use alloc::{collections::BTreeSet, vec::Vec};
use core::{
    alloc::Layout,
    array,
    cmp::{max, min},
    ops::{Deref, DerefMut, Range},
    sync::atomic::{AtomicBool, Ordering::SeqCst},
};

use buddy_system_allocator::FrameAllocator;
use local_alloc_macro::local_alloc_disabled;

use crate::{
    cpu_local, sync::SpinLock, task::PerTaskMap, vm::hierarchical_heap_allocator::prev_power_of_two,
};

pub(super) struct HierarchicalFrameAllocator<const ORDER: usize = 32> {
    global_frame_allocator: SpinLock<GlobalFrameAllocator<ORDER>>,
    local_frame_allocators: PerTaskMap<LocalFrameAllocator<ORDER>>,
}

impl<const ORDER: usize> HierarchicalFrameAllocator<ORDER> {
    pub fn new() -> Self {
        Self {
            global_frame_allocator: SpinLock::new(GlobalFrameAllocator::<ORDER>::new()),
            local_frame_allocators: PerTaskMap::new(),
        }
    }

    /// Register the available frames to the global allocator
    pub fn add_frame(&self, start: usize, end: usize) {
        self.global_frame_allocator.lock().add_frame(start, end);
    }

    /// Choose whether to allocate memory from local or global allocator based on the current
    /// task ID and LOCAL_ALLOC_ENABLED tag.
    pub fn alloc(&self, count: usize) -> Option<usize> {
        if is_local_alloc_enabled() {
            if !self.local_frame_allocators.local_exists() {
                self.local_frame_allocators
                    .local_insert(LocalFrameAllocator::new());
            }
            return self.local_frame_allocators.local_mut().alloc(count);
        }
        self.global_frame_allocator.lock().alloc(count)
    }

    /// Choose whether to deallocate memory from local or global allocator based on the current
    /// task ID and LOCAL_ALLOC_ENABLED tag.
    ///
    /// The allocators selected by alloc and dealloc must be the same, if they are not, an error
    /// will be reported in the corresponding dealloc function
    pub fn dealloc(&self, start_frame: usize, count: usize) {
        if is_local_alloc_enabled() {
            return self
                .local_frame_allocators
                .local_mut()
                .dealloc(start_frame, count);
        }

        self.global_frame_allocator
            .lock()
            .dealloc(start_frame, count);
    }

    pub fn destory_local(&self) {
        let local_allocator = self.local_frame_allocators.local();
        assert!(local_allocator.allocated() == 0);

        for (start_frame, size) in local_allocator.free_list() {
            self.global_frame_allocator
                .lock()
                .dealloc(start_frame, size);
        }
        self.local_frame_allocators.remove();
    }

    fn assign_frame(&self, size: usize) {
        let size = size.next_power_of_two();
        let start_frame = self.global_frame_allocator.lock().alloc(size).unwrap();
        self.local_frame_allocators
            .local_mut()
            .add_frame(start_frame, start_frame + size);
    }

    fn recall_frame(&self, size: usize) {
        let size = size.next_power_of_two();
        let start_frame = self
            .local_frame_allocators
            .local_mut()
            .delete_frame(size)
            .unwrap();

        let start_frame = self
            .global_frame_allocator
            .lock()
            .dealloc(start_frame, size);
    }
}

// The `HierarchicalFrameAllocator` has the `Sync` attribute because different
// tasks are indexed to their corresponding local heaps and do not compete with each other.
unsafe impl<const ORDER: usize> Sync for HierarchicalFrameAllocator<ORDER> {}

struct GlobalFrameAllocator<const ORDER: usize = 32> {
    allocator: FrameAllocator<ORDER>,
}

impl<const ORDER: usize> GlobalFrameAllocator<ORDER> {
    pub fn new() -> Self {
        Self {
            allocator: FrameAllocator::<ORDER>::new(),
        }
    }
}

impl<const ORDER: usize> Deref for GlobalFrameAllocator<ORDER> {
    type Target = FrameAllocator<ORDER>;
    fn deref(&self) -> &Self::Target {
        &self.allocator
    }
}

impl<const ORDER: usize> DerefMut for GlobalFrameAllocator<ORDER> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.allocator
    }
}

struct LocalFrameAllocator<const ORDER: usize = 32> {
    // buddy system with max order of ORDER
    free_list: [BTreeSet<usize>; ORDER],

    // statistics
    allocated: usize,
    total: usize,
}

impl<const ORDER: usize> LocalFrameAllocator<ORDER> {
    /// Create an empty frame allocator
    pub fn new() -> Self {
        Self {
            free_list: array::from_fn(|_| BTreeSet::default()),
            allocated: 0,
            total: 0,
        }
    }

    /// Add a range of frame number [start, end) to the allocator,
    /// these frames are allocated from the global allocator
    pub fn add_frame(&mut self, start: usize, end: usize) {
        assert!(start <= end);

        let mut total = 0;
        let mut current_start = start;

        while current_start < end {
            let lowbit = if current_start > 0 {
                current_start & (!current_start + 1)
            } else {
                32
            };
            let size = min(
                min(lowbit, prev_power_of_two(end - current_start)),
                1 << (ORDER - 1),
            );
            total += size;

            self.free_list[size.trailing_zeros() as usize].insert(current_start);
            current_start += size;
        }

        self.total += total;
    }

    pub fn delete_frame(&mut self, size: usize) -> Option<usize> {
        let class = size.trailing_zeros() as usize;

        for i in class..self.free_list.len() {
            if !self.free_list[i].is_empty() {
                for j in (class + 1..i + 1).rev() {
                    if let Some(block_ref) = self.free_list[j].iter().next() {
                        let block = *block_ref;
                        self.free_list[j - 1].insert(block + (1 << (j - 1)));
                        self.free_list[j - 1].insert(block);
                        self.free_list[j].remove(&block);
                    } else {
                        return None;
                    }
                }

                let result = self.free_list[class].iter().next().clone();
                if let Some(result_ref) = result {
                    let result = *result_ref;
                    self.free_list[class].remove(&result);
                    self.total -= size;
                    return Some(result);
                }
            }
        }
        None
    }

    /// Add a range of frames to the allocator.
    fn insert(&mut self, range: Range<usize>) {
        self.add_frame(range.start, range.end);
    }

    /// Allocate a range of frames from the allocator, returning the first frame of the allocated
    /// range.
    pub fn alloc(&mut self, count: usize) -> Option<usize> {
        let size = count.next_power_of_two();
        self.alloc_power_of_two(size)
    }

    /// Allocate a range of frames with the given size and alignment from the allocator, returning
    /// the first frame of the allocated range.
    pub fn alloc_aligned(&mut self, layout: Layout) -> Option<usize> {
        let size = max(layout.size().next_power_of_two(), layout.align());
        self.alloc_power_of_two(size)
    }

    /// Allocate a range of frames of the given size from the allocator. The size must be a power of
    /// two. The allocated range will have alignment equal to the size.
    fn alloc_power_of_two(&mut self, size: usize) -> Option<usize> {
        let class = size.trailing_zeros() as usize;
        for i in class..self.free_list.len() {
            // Find the first non-empty size class
            if !self.free_list[i].is_empty() {
                // Split buffers
                for j in (class + 1..i + 1).rev() {
                    if let Some(block_ref) = self.free_list[j].iter().next() {
                        let block = *block_ref;
                        self.free_list[j - 1].insert(block + (1 << (j - 1)));
                        self.free_list[j - 1].insert(block);
                        self.free_list[j].remove(&block);
                    } else {
                        return None;
                    }
                }

                let result = self.free_list[class].iter().next().clone();
                if let Some(result_ref) = result {
                    let result = *result_ref;
                    self.free_list[class].remove(&result);
                    self.allocated += size;
                    return Some(result);
                }
            }
        }
        None
    }

    /// Deallocate a range of frames [frame, frame+count) from the frame allocator.
    ///
    /// The range should be exactly the same when it was allocated, as in heap allocator
    pub fn dealloc(&mut self, start_frame: usize, count: usize) {
        let size = count.next_power_of_two();
        self.dealloc_power_of_two(start_frame, size)
    }

    /// Deallocate a range of frames which was previously allocated by [`alloc_aligned`].
    ///
    /// The layout must be exactly the same as when it was allocated.
    pub fn dealloc_aligned(&mut self, start_frame: usize, layout: Layout) {
        let size = max(layout.size().next_power_of_two(), layout.align());
        self.dealloc_power_of_two(start_frame, size)
    }

    /// Deallocate a range of frames with the given size from the allocator. The size must be a
    /// power of two.
    fn dealloc_power_of_two(&mut self, start_frame: usize, size: usize) {
        let class = size.trailing_zeros() as usize;

        // Merge free buddy lists
        let mut current_ptr = start_frame;
        let mut current_class = class;
        while current_class < self.free_list.len() {
            let buddy = current_ptr ^ (1 << current_class);
            if self.free_list[current_class].remove(&buddy) == true {
                // Free buddy found
                current_ptr = min(current_ptr, buddy);
                current_class += 1;
            } else {
                self.free_list[current_class].insert(current_ptr);
                break;
            }
        }

        self.allocated -= size;
    }

    pub fn allocated(&self) -> usize {
        self.allocated
    }

    pub fn free_list(&self) -> Vec<(usize, usize)> {
        let mut frames = Vec::new();
        for i in 0..self.free_list.len() {
            for start_frame in self.free_list[i].iter() {
                frames.push((*start_frame, (1 << i) as usize));
            }
        }
        frames
    }
}

pub fn is_local_alloc_enabled() -> bool {
    LOCAL_ALLOC_ENABLED.load(SeqCst)
}

cpu_local! {
    static LOCAL_ALLOC_ENABLED: AtomicBool = AtomicBool::new(true);
}

pub struct LocalAllocGuard {
    old_value: bool,
}

impl LocalAllocGuard {
    fn new() -> Self {
        LocalAllocGuard {
            old_value: LOCAL_ALLOC_ENABLED.fetch_and(false, SeqCst),
        }
    }
}

impl Drop for LocalAllocGuard {
    fn drop(&mut self) {
        LOCAL_ALLOC_ENABLED.store(self.old_value, SeqCst);
    }
}
