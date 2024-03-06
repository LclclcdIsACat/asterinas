use core::{
    alloc::{GlobalAlloc, Layout},
    cmp::{max, min},
    mem::size_of,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::Ordering::SeqCst,
};

use align_ext::AlignExt;
use buddy_system_allocator::{linked_list, Heap};

use crate::{
    config::{KERNEL_HEAP_SIZE, PAGE_SIZE},
    prelude::*,
    sync::SpinLock,
    task::PerTaskMap,
    trap::disable_local,
    vm::{
        frame_allocator::FRAME_ALLOCATOR, hierarchical_frame_allocator::LOCAL_ALLOC_ENABLED,
        paddr_to_vaddr, vaddr_to_paddr,
    },
    Error,
};

#[global_allocator]
static HEAP_ALLOCATOR: HierarchicalHeapAllocator<32> = HierarchicalHeapAllocator::new();

#[alloc_error_handler]
pub fn handle_alloc_error(layout: core::alloc::Layout) -> ! {
    panic!("Heap allocation error, layout = {:?}", layout);
}

static mut HEAP_SPACE: [u8; KERNEL_HEAP_SIZE] = [0; KERNEL_HEAP_SIZE];

pub fn init() {
    // Safety: The HEAP_SPACE is a static memory range, so it's always valid.
    unsafe {
        HEAP_ALLOCATOR.init(HEAP_SPACE.as_ptr(), KERNEL_HEAP_SIZE);
    }
}

struct HierarchicalHeapAllocator<const ORDER: usize> {
    global_heap: SpinLock<GlobalHeapAllocator<ORDER>>,
    local_heap: PerTaskMap<LocalHeapAllocator<ORDER>>,
}

impl<const ORDER: usize> HierarchicalHeapAllocator<ORDER> {
    pub const fn new() -> Self {
        Self {
            global_heap: SpinLock::new(GlobalHeapAllocator::<ORDER>::new()),
            local_heap: PerTaskMap::new(),
        }
    }

    pub unsafe fn init(&self, start: *const u8, size: usize) {
        self.global_heap
            .lock_irq_disabled()
            .init(start as usize, size);
    }

    fn rescue(&self, layout: &Layout) -> Result<()> {
        // TODO: Optimize the allocation method with fixed threshold
        const MIN_NUM_LOCAL_FRAMES: usize = 0x800000 / PAGE_SIZE; // 8MB

        let mut num_frames = {
            let align = PAGE_SIZE.max(layout.align());
            debug_assert!(align % PAGE_SIZE == 0);
            let size = layout.size().align_up(align);
            size / PAGE_SIZE
        };

        let frame_allocator = FRAME_ALLOCATOR.get().unwrap();

        let allocation_start = {
            if num_frames >= MIN_NUM_LOCAL_FRAMES {
                frame_allocator.alloc(num_frames).ok_or(Error::NoMemory)?
            } else {
                match frame_allocator.alloc(MIN_NUM_LOCAL_FRAMES) {
                    None => frame_allocator.alloc(num_frames).ok_or(Error::NoMemory)?,
                    Some(start) => {
                        num_frames = MIN_NUM_LOCAL_FRAMES;
                        start
                    }
                }
            }
        };

        // FIXME: the alloc function internally allocates heap memory(inside FrameAllocator).
        // So if the heap is nearly run out, allocating frame will fail too.
        let vaddr = paddr_to_vaddr(allocation_start * PAGE_SIZE);

        // local heap
        if LOCAL_ALLOC_ENABLED.load(SeqCst) {
            unsafe {
                self.local_heap
                    .local_mut()
                    .add_to_heap(vaddr, PAGE_SIZE * num_frames);
            }
        }

        // global heap
        unsafe {
            self.global_heap
                .lock_irq_disabled()
                .add_to_heap(vaddr, PAGE_SIZE * num_frames);
        }
        Ok(())
    }
}

unsafe impl<const ORDER: usize> GlobalAlloc for HierarchicalHeapAllocator<ORDER> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _guard = disable_local();
        if LOCAL_ALLOC_ENABLED.load(SeqCst) {
            if !self.local_heap.local_exists() {
                self.local_heap.local_insert(LocalHeapAllocator::new());
            }
            let local_heap = self.local_heap.local_mut();
            if let Ok(allocation) = local_heap.alloc(layout) {
                return allocation.as_ptr();
            }

            // Avoid locking self.heap when calling rescue.
            if self.rescue(&layout).is_err() {
                return core::ptr::null_mut::<u8>();
            }

            return self
                .local_heap
                .local_mut()
                .alloc(layout)
                .map_or(core::ptr::null_mut::<u8>(), |allocation| {
                    allocation.as_ptr()
                });
        }

        if let Ok(allocation) = self.global_heap.lock().alloc(layout) {
            return allocation.as_ptr();
        }

        // Avoid locking self.heap when calling rescue.
        if self.rescue(&layout).is_err() {
            return core::ptr::null_mut::<u8>();
        }

        self.global_heap
            .lock()
            .alloc(layout)
            .map_or(core::ptr::null_mut::<u8>(), |allocation| {
                allocation.as_ptr()
            })
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if LOCAL_ALLOC_ENABLED.load(SeqCst) {
            self.local_heap
                .local_mut()
                .dealloc(NonNull::new_unchecked(ptr), layout);
        } else {
            self.global_heap
                .lock_irq_disabled()
                .dealloc(NonNull::new_unchecked(ptr), layout);
        }
    }
}

// The `HierarchicalHeapAllocator` has the `Sync` attribute because different
// tasks are indexed to their corresponding local heaps and do not compete with each other.
unsafe impl<const ORDER: usize> Sync for HierarchicalHeapAllocator<ORDER> {}

struct GlobalHeapAllocator<const ORDER: usize> {
    heap: Heap<ORDER>,
}

impl<const ORDER: usize> GlobalHeapAllocator<ORDER> {
    pub const fn new() -> Self {
        Self {
            heap: Heap::<ORDER>::new(),
        }
    }
}

impl<const ORDER: usize> Deref for GlobalHeapAllocator<ORDER> {
    type Target = Heap<ORDER>;
    fn deref(&self) -> &Self::Target {
        &self.heap
    }
}

impl<const ORDER: usize> DerefMut for GlobalHeapAllocator<ORDER> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.heap
    }
}

struct LocalHeapAllocator<const ORDER: usize> {
    // buddy system with max order of `ORDER`
    free_list: [linked_list::LinkedList; ORDER],

    // statistics
    user: usize,
    allocated: usize,
    total: usize,
}

impl<const ORDER: usize> LocalHeapAllocator<ORDER> {
    /// Create an empty heap
    pub const fn new() -> Self {
        Self {
            free_list: [linked_list::LinkedList::new(); ORDER],
            user: 0,
            allocated: 0,
            total: 0,
        }
    }

    /// Add a range of memory [start, end) to the heap
    pub unsafe fn add_to_heap(&mut self, mut start: usize, mut end: usize) {
        // avoid unaligned access on some platforms
        start = (start + size_of::<usize>() - 1) & (!size_of::<usize>() + 1);
        end = end & (!size_of::<usize>() + 1);
        assert!(start <= end);

        let mut total = 0;
        let mut current_start = start;

        while current_start + size_of::<usize>() <= end {
            let lowbit = current_start & (!current_start + 1);
            let size = min(lowbit, prev_power_of_two(end - current_start));
            total += size;

            self.free_list[size.trailing_zeros() as usize].push(current_start as *mut usize);
            current_start += size;
        }

        self.total += total;
    }

    // FIXEM: add for a reasonable local memory release mechanism
    pub unsafe fn delete_from_heap(&mut self, size: usize) {
        let class = size.trailing_zeros() as usize;
        for i in class..self.free_list.len() {
            if !self.free_list[i].is_empty() {
                for j in (class + 1..i + 1).rev() {
                    if let Some(block) = self.free_list[j].pop() {
                        unsafe {
                            self.free_list[j - 1]
                                .push((block as usize + (1 << (j - 1))) as *mut usize);
                            self.free_list[j - 1].push(block);
                        }
                    } else {
                        return;
                    }
                }
                let result = self.free_list[class].pop().unwrap() as usize;
                let paddr = vaddr_to_paddr(result).unwrap();
                FRAME_ALLOCATOR.get().unwrap().dealloc(paddr, size);
                self.total -= size;
                return;
            }
        }
    }

    pub unsafe fn clear(&mut self) {
        for i in 0..self.free_list.len() {
            while !self.free_list[i].is_empty() {
                let result = self.free_list[i].pop().unwrap() as usize;
                let paddr = vaddr_to_paddr(result).unwrap();
                FRAME_ALLOCATOR.get().unwrap().dealloc(paddr, 1 << i);
                self.total -= 1 << i;
            }
        }
        assert!(self.total == 0);
    }

    /// Add a range of memory [start, start+size) to the heap
    pub unsafe fn init(&mut self, start: usize, size: usize) {
        self.add_to_heap(start, start + size);
    }

    /// Alloc a range of memory from the heap satifying `layout` requirements
    pub fn alloc(&mut self, layout: Layout) -> core::result::Result<NonNull<u8>, ()> {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;
        for i in class..self.free_list.len() {
            // Find the first non-empty size class
            if !self.free_list[i].is_empty() {
                // Split buffers
                for j in (class + 1..i + 1).rev() {
                    if let Some(block) = self.free_list[j].pop() {
                        unsafe {
                            self.free_list[j - 1]
                                .push((block as usize + (1 << (j - 1))) as *mut usize);
                            self.free_list[j - 1].push(block);
                        }
                    } else {
                        return Err(());
                    }
                }

                let result = NonNull::new(
                    self.free_list[class]
                        .pop()
                        .expect("current block should have free space now")
                        as *mut u8,
                );
                if let Some(result) = result {
                    self.user += layout.size();
                    self.allocated += size;
                    return Ok(result);
                } else {
                    return Err(());
                }
            }
        }
        Err(())
    }

    /// Dealloc a range of memory from the heap
    pub fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        unsafe {
            // Put back into free list
            self.free_list[class].push(ptr.as_ptr() as *mut usize);

            // Merge free buddy lists
            let mut current_ptr = ptr.as_ptr() as usize;
            let mut current_class = class;
            while current_class < self.free_list.len() {
                let buddy = current_ptr ^ (1 << current_class);
                let mut flag = false;
                for block in self.free_list[current_class].iter_mut() {
                    if block.value() as usize == buddy {
                        block.pop();
                        flag = true;
                        break;
                    }
                }

                // Free buddy found
                if flag {
                    self.free_list[current_class].pop();
                    current_ptr = min(current_ptr, buddy);
                    current_class += 1;
                    self.free_list[current_class].push(current_ptr as *mut usize);
                } else {
                    break;
                }
            }
        }

        self.user -= layout.size();
        self.allocated -= size;
    }

    /// Return the number of bytes that user requests
    pub fn stats_alloc_user(&self) -> usize {
        self.user
    }

    /// Return the number of bytes that are actually allocated
    pub fn stats_alloc_actual(&self) -> usize {
        self.allocated
    }

    /// Return the total number of bytes in the heap
    pub fn stats_total_bytes(&self) -> usize {
        self.total
    }
}

pub(super) fn prev_power_of_two(num: usize) -> usize {
    1 << (8 * (size_of::<usize>()) - num.leading_zeros() as usize - 1)
}
