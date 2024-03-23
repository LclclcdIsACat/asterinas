// SPDX-License-Identifier: MPL-2.0

use core::ops::Range;

use align_ext::AlignExt;
use aster_frame::vm::{VmAllocOptions, VmFrame, VmFrameVec, VmIo, VmMapOptions, VmPerm, VmSpace};

use super::{get_intersected_range, interval::Interval, is_intersected, Vmar, Vmar_};
use crate::{
    prelude::*,
    vm::{get_aligned_addr_range, get_page_idx_range, pager::Pager, perms::VmPerms, vmar::Rights},
};

/// A VmMapping represents mapping a vmo into a vmar.
/// A vmar can has multiple VmMappings, which means multiple vmos are mapped to a vmar.
/// A vmo can also contain multiple VmMappings, which means a vmo can be mapped to multiple vmars.
/// The reltionship between Vmar and Vmo is M:N.
pub struct VmMapping {
    inner: Mutex<VmMappingInner>,
    backend: Option<VmMappingBackend>,
    /// The parent vmar. The parent should always point to a valid vmar.
    parent: Weak<Vmar_>,
}

impl VmMapping {
    pub fn try_clone(&self) -> Result<Self> {
        let inner = self.inner.lock().clone();
        Ok(Self {
            inner: Mutex::new(inner),
            backend: self.backend.clone(),
            parent: self.parent.clone(),
        })
    }
}

#[derive(Clone)]
struct VmMappingInner {
    /// The size of mapping, in bytes. The map size can even be larger than the size of vmo.
    /// Those pages outside vmo range cannot be read or write.
    map_size: usize,
    /// The base address relative to the root vmar where the vmo is mapped.
    map_to_addr: Vaddr,
    /// The pages already mapped. The key is the vaddr.
    /// TODO: it is more reasonable to index using virtual page frame numbers(VPNs) here.
    /// However, we currently do not have an abstraction of VPNs.
    mapped_pages: BTreeSet<Vaddr>,
    /// The permission of each page. The key is the vaddr.
    /// This map can be filled when mapping a vmo to vmar and can be modified when call mprotect.
    /// We keep the options in case the page is not committed(or create copy on write mappings) and will further need these options.
    /// TODO: it is more reasonable to index using virtual page frame numbers(VPNs) here.
    /// However, we currently do not have an abstraction of VPNs.
    page_perms: BTreeMap<Vaddr, VmPerm>,

    flag: VmMappingFlags,
}

bitflags! {
    /// Vm_mapping flags.
    pub struct VmMappingFlags: u32 {
        /// Set this flag if a mapping is destroyed.
        const DESTROYED  = 1 << 0;

        /// Set this flag if a mapping is a copy-on-write mapping.
        const COW = 1 << 1;
    }
}

impl Interval<usize> for Arc<VmMapping> {
    fn range(&self) -> Range<usize> {
        self.map_to_addr()..self.map_to_addr() + self.map_size()
    }
}

impl VmMapping {
    pub fn build_mapping<R1>(option: VmarMapOptions<R1>) -> Result<Self> {
        let VmarMapOptions {
            parent,
            perms,
            backend_offset,
            size,
            offset,
            align,
            can_overwrite,
            pager,
        } = option;
        let Vmar(parent_vmar, _) = parent;
        let map_to_addr =
            parent_vmar.allocate_free_region_for_vmo(size, offset, align, can_overwrite)?;
        trace!(
            "build mapping, map_range = 0x{:x}- 0x{:x}",
            map_to_addr,
            map_to_addr + size
        );

        let page_perms = {
            let mut page_perms = BTreeMap::new();
            let perm = VmPerm::from(perms);
            let map_addr_range = get_aligned_addr_range(&(map_to_addr..map_to_addr + size));
            for map_addr in map_addr_range.step_by(PAGE_SIZE) {
                page_perms.insert(map_addr, perm);
            }
            page_perms
        };

        let vm_mapping_inner = VmMappingInner {
            map_size: size,
            map_to_addr,
            mapped_pages: BTreeSet::new(),
            page_perms,
            flag: VmMappingFlags::empty(),
        };

        let backend = {
            if let Some(pager) = pager {
                Some(VmMappingBackend::new(pager, backend_offset))
            } else {
                None
            }
        };

        Ok(Self {
            inner: Mutex::new(vm_mapping_inner),
            parent: Arc::downgrade(&parent_vmar),
            backend,
        })
    }

    /// Add a new committed page and map it to vmspace. If copy on write is set, it's allowed to unmap the page at the same address.
    /// FIXME: This implementation based on the truth that we map one page at a time. If multiple pages are mapped together, this implementation may have problems
    pub(super) fn map_one_page(
        &self,
        map_addr: Vaddr,
        frame: VmFrame,
        is_readonly: bool,
    ) -> Result<()> {
        let parent = self.parent.upgrade().unwrap();
        let vm_space = parent.vm_space();
        self.inner
            .lock()
            .map_one_page(vm_space, map_addr, frame, is_readonly)
    }

    /// unmap a page
    pub(super) fn unmap_one_page(&self, map_addr: Vaddr) -> Result<()> {
        let parent = self.parent.upgrade().unwrap();
        let vm_space = parent.vm_space();
        self.inner.lock().unmap_one_page(vm_space, map_addr)
    }

    /// the mapping's start address
    pub fn map_to_addr(&self) -> Vaddr {
        self.inner.lock().map_to_addr
    }

    /// the mapping's size
    pub fn map_size(&self) -> usize {
        self.inner.lock().map_size
    }

    pub fn can_read_bytes(&self, read_addr: Vaddr, len: usize) -> Result<()> {
        // TODO: the current logic is vulnerable to TOCTTOU attack, since the permission may change after check.
        let aligned_addr_range = get_aligned_addr_range(&(read_addr..read_addr + len));
        let read_perm = VmPerm::R;
        for page_addr in aligned_addr_range.step_by(PAGE_SIZE) {
            self.check_perm(&page_addr, &read_perm)?;
        }
        Ok(())
    }

    pub fn can_write_bytes(&self, write_addr: Vaddr, len: usize) -> Result<()> {
        let aligned_addr_range = get_aligned_addr_range(&(write_addr..write_addr + len));
        let write_perm = VmPerm::W;
        for page_addr in aligned_addr_range.step_by(PAGE_SIZE) {
            self.check_perm(&page_addr, &write_perm)?;
        }
        Ok(())
    }

    /// Unmap pages in the range
    pub fn unmap(&self, addr_range: &Range<usize>, may_destroy: bool) -> Result<()> {
        let addr_range = get_intersected_range(addr_range, &self.range());
        let parent = self.parent.upgrade().unwrap();
        let vm_space = parent.vm_space();
        self.inner.lock().unmap(vm_space, &addr_range, may_destroy)
    }

    pub fn unmap_and_decommit(&self, addr_range: Range<usize>) -> Result<()> {
        let addr_range = get_intersected_range(&addr_range, &self.range());
        self.unmap(&addr_range, false)?;
        if let Some(backend) = self.backend {
            let decommit_range = {
                let map_to_addr = self.map_to_addr();
                let backend_offset = backend.offset();
                (addr_range.start - map_to_addr + backend_offset)
                    ..(addr_range.end - map_to_addr + backend_offset)
            };
            let page_idx_range = get_page_idx_range(&decommit_range);
            for page_idx in page_idx_range {
                backend.pager().decommit_page(page_idx);
            }
        };
        Ok(())
    }

    pub fn is_destroyed(&self) -> bool {
        self.inner.lock().flag.contains(VmMappingFlags::DESTROYED)
    }

    pub fn handle_page_fault(
        &self,
        page_fault_addr: Vaddr,
        not_present: bool,
        write: bool,
    ) -> Result<()> {
        let map_offset = page_fault_addr - self.map_to_addr();
        let aligned_fault_addr = page_fault_addr.align_down(PAGE_SIZE);
        if map_offset >= self.map_size() {
            return_errno_with_message!(Errno::EACCES, "page fault addr is not mapped.");
        }
        if write {
            self.check_rights(Rights::WRITE)?;
        } else {
            self.check_rights(Rights::READ)?;
        }

        let required_perm = if write { VmPerm::W } else { VmPerm::R };
        self.check_perm(&aligned_fault_addr, &required_perm)?;

        let frame = {
            match &self.backend {
                None => VmAllocOptions::new(1).alloc_single()?,
                Some(backend) => backend
                    .pager()
                    .commit_page((backend.offset() + map_offset) / PAGE_SIZE)?,
            }
        };

        // If read access to cow vmo triggers page fault, the map should be readonly.
        // If user next tries to write to the frame, another page fault will be triggered.
        let is_readonly = self.inner.lock().flag.contains(VmMappingFlags::COW) && !write;
        self.map_one_page(aligned_fault_addr, frame, is_readonly)
    }

    pub(super) fn protect(&self, perms: VmPerms, range: Range<usize>) -> Result<()> {
        let rights = Rights::from(perms);
        self.check_rights(rights)?;
        let vmar = self.parent.upgrade().unwrap();
        let vm_space = vmar.vm_space();
        self.inner.lock().protect(vm_space, perms, range)
    }

    pub(super) fn new_cow(&self, new_parent: &Arc<Vmar_>) -> Result<VmMapping> {
        let VmMapping { inner, backend, .. } = self;

        let new_inner = {
            let mut inner = self.inner.lock();
            inner.flag |= VmMappingFlags::COW;
            VmMappingInner {
                map_size: inner.map_size,
                map_to_addr: inner.map_to_addr,
                mapped_pages: BTreeSet::new(),
                page_perms: inner.page_perms.clone(),
                flag: inner.flag,
            }
        };

        let vmar = self.parent.upgrade().unwrap();
        let vm_space = vmar.vm_space();

        Ok(VmMapping {
            inner: Mutex::new(new_inner),
            backend: backend.clone(),
            parent: Arc::downgrade(new_parent),
        })
    }

    pub fn range(&self) -> Range<usize> {
        self.map_to_addr()..self.map_to_addr() + self.map_size()
    }

    /// Trim a range from the mapping.
    /// There are several cases.
    /// 1. the trim_range is totally in the mapping. Then the mapping will split as two mappings.
    /// 2. the trim_range covers the mapping. Then the mapping will be destroyed.
    /// 3. the trim_range partly overlaps with the mapping, in left or right. Only overlapped part is trimmed.
    /// If we create a mapping with a new map addr, we will add it to mappings_to_append.
    /// If the mapping with map addr does not exist ever, the map addr will be added to mappings_to_remove.
    /// Otherwise, we will directly modify self.
    pub fn trim_mapping(
        self: &Arc<Self>,
        trim_range: &Range<usize>,
        mappings_to_remove: &mut BTreeSet<Vaddr>,
        mappings_to_append: &mut BTreeMap<Vaddr, Arc<VmMapping>>,
    ) -> Result<()> {
        let map_to_addr = self.map_to_addr();
        let map_size = self.map_size();
        let range = self.range();
        if !is_intersected(&range, trim_range) {
            return Ok(());
        }
        if trim_range.start <= map_to_addr && trim_range.end >= map_to_addr + map_size {
            // fast path: the whole mapping was trimed
            self.unmap(trim_range, true)?;
            mappings_to_remove.insert(map_to_addr);
            return Ok(());
        }
        if trim_range.start <= range.start {
            mappings_to_remove.insert(map_to_addr);
            if trim_range.end <= range.end {
                // overlap vm_mapping from left
                let new_map_addr = self.trim_left(trim_range.end)?;
                mappings_to_append.insert(new_map_addr, self.clone());
            } else {
                // the mapping was totally destroyed
            }
        } else {
            if trim_range.end <= range.end {
                // the trim range was totally inside the old mapping
                let another_mapping = Arc::new(self.try_clone()?);
                let another_map_to_addr = another_mapping.trim_left(trim_range.end)?;
                mappings_to_append.insert(another_map_to_addr, another_mapping);
            } else {
                // overlap vm_mapping from right
            }
            self.trim_right(trim_range.start)?;
        }

        Ok(())
    }

    /// trim the mapping from left to a new address.
    fn trim_left(&self, vaddr: Vaddr) -> Result<Vaddr> {
        let vmar = self.parent.upgrade().unwrap();
        let vm_space = vmar.vm_space();
        self.inner.lock().trim_left(vm_space, vaddr)
    }

    /// trim the mapping from right to a new address.
    fn trim_right(&self, vaddr: Vaddr) -> Result<Vaddr> {
        let vmar = self.parent.upgrade().unwrap();
        let vm_space = vmar.vm_space();
        self.inner.lock().trim_right(vm_space, vaddr)
    }

    fn check_perm(&self, page_idx: &usize, perm: &VmPerm) -> Result<()> {
        self.inner.lock().check_perm(page_idx, perm)
    }

    fn check_rights(&self, rights: Rights) -> Result<()> {
        match &self.pager {
            None => Ok(()),
            Some(pager) => pager.check_rights(rights),
        }
    }
}

impl VmMappingInner {
    fn map_one_page(
        &mut self,
        vm_space: &VmSpace,
        map_addr: Vaddr,
        frame: VmFrame,
        is_readonly: bool,
    ) -> Result<()> {
        let map_addr = map_addr.align_down(PAGE_SIZE);

        let vm_perm = {
            let mut perm = *self.page_perms.get(&page_idx).unwrap();
            if is_readonly {
                perm -= VmPerm::W;
            }
            perm
        };

        let vm_map_options = {
            let mut options = VmMapOptions::new();
            options.addr(Some(map_addr));
            options.perm(vm_perm);
            options
        };

        // cow child allows unmapping the mapped page
        if self.flag.contains(VmMappingFlags::COW) && vm_space.is_mapped(map_addr) {
            vm_space.unmap(&(map_addr..(map_addr + PAGE_SIZE))).unwrap();
        }

        vm_space.map(VmFrameVec::from_one_frame(frame), &vm_map_options)?;
        self.mapped_pages.insert(page_idx);
        Ok(())
    }

    fn unmap_one_page(&mut self, vm_space: &VmSpace, map_addr: Vaddr) -> Result<()> {
        let map_addr = map_addr.align_down(PAGE_SIZE);
        let range = map_addr..(map_addr + PAGE_SIZE);
        if vm_space.is_mapped(map_addr) {
            vm_space.unmap(&range)?;
        }
        self.mapped_pages.remove(&map_addr);
        Ok(())
    }

    /// Unmap pages in the range
    fn unmap(
        &mut self,
        vm_space: &VmSpace,
        addr_range: &Range<usize>,
        may_destroy: bool,
    ) -> Result<()> {
        let map_start = addr_range.start.align_down(PAGE_SIZE);
        let map_end = addr_range.end.align_up(PAGE_SIZE);

        for map_addr in (map_start..map_end).step_by(PAGE_SIZE) {
            self.unmap_one_page(vm_space, map_addr)?;
        }
        if may_destroy && *addr_range == self.addr_range() {
            self.flag.remove(VmMappingFlags::DESTROYED);
        }
        Ok(())
    }

    pub(super) fn protect(
        &mut self,
        vm_space: &VmSpace,
        perms: VmPerms,
        range: Range<usize>,
    ) -> Result<()> {
        debug_assert!(range.start % PAGE_SIZE == 0);
        debug_assert!(range.end % PAGE_SIZE == 0);
        let addr_range = get_aligned_addr_range(&range);
        let perm = VmPerm::from(perms);
        for addr in addr_range {
            self.page_perms.insert(addr, perm);
            if vm_space.is_mapped(page_addr) {
                // if the page is already mapped, we will modify page table
                let perm = VmPerm::from(perms);
                let page_range = page_addr..(page_addr + PAGE_SIZE);
                vm_space.protect(&page_range, perm)?;
            }
        }
        Ok(())
    }

    /// trim the mapping from left to a new address.
    fn trim_left(&mut self, vm_space: &VmSpace, vaddr: Vaddr) -> Result<Vaddr> {
        trace!(
            "trim left: range: {:x?}, vaddr = 0x{:x}",
            self.range(),
            vaddr
        );
        debug_assert!(vaddr >= self.map_to_addr && vaddr <= self.map_to_addr + self.map_size);
        debug_assert!(vaddr % PAGE_SIZE == 0);
        let trim_size = vaddr - self.map_to_addr;

        self.map_to_addr = vaddr;
        let old_vmo_offset = self.vmo_offset;
        self.vmo_offset += trim_size;
        self.map_size -= trim_size;
        for page_idx in old_vmo_offset / PAGE_SIZE..self.vmo_offset / PAGE_SIZE {
            self.page_perms.remove(&page_idx);
            if self.mapped_pages.remove(&page_idx) {
                let _ = self.unmap_one_page(vm_space, page_idx);
            }
        }
        Ok(self.map_to_addr)
    }

    /// trim the mapping from right to a new address.
    fn trim_right(&mut self, vm_space: &VmSpace, vaddr: Vaddr) -> Result<Vaddr> {
        trace!(
            "trim right: range: {:x?}, vaddr = 0x{:x}",
            self.range(),
            vaddr
        );
        debug_assert!(vaddr >= self.map_to_addr && vaddr <= self.map_to_addr + self.map_size);
        debug_assert!(vaddr % PAGE_SIZE == 0);
        let page_idx_range = (vaddr - self.map_to_addr + self.vmo_offset) / PAGE_SIZE
            ..(self.map_size + self.vmo_offset) / PAGE_SIZE;
        for page_idx in page_idx_range {
            self.page_perms.remove(&page_idx);
            let _ = self.unmap_one_page(vm_space, page_idx);
        }
        self.map_size = vaddr - self.map_to_addr;
        Ok(self.map_to_addr)
    }

    fn range(&self) -> Range<usize> {
        self.map_to_addr..self.map_to_addr + self.map_size
    }

    fn check_perm(&self, page_idx: &usize, perm: &VmPerm) -> Result<()> {
        let page_perm = self
            .page_perms
            .get(page_idx)
            .ok_or(Error::with_message(Errno::EINVAL, "invalid page idx"))?;

        if !page_perm.contains(*perm) {
            return_errno_with_message!(Errno::EACCES, "perm check fails");
        }

        Ok(())
    }
}

#[derive(Clone)]
struct VmMappingBackend {
    pager: Arc<dyn Pager>,
    offset: usize,
}

impl VmMappingBackend {
    pub fn new(pager: Arc<dyn Pager>, offset: usize) -> Self {
        Self { pager, offset }
    }

    pub fn pager(&self) -> Arc<dyn Pager> {
        self.pager
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// Options for creating a new mapping. The mapping is not allowed to overlap
/// with any child VMARs. And unless specified otherwise, it is not allowed
/// to overlap with any existing mapping, either.
pub struct VmarMapOptions<R> {
    parent: Vmar<R>,
    perms: VmPerms,
    backend_offset: usize,
    size: usize,
    offset: Option<usize>,
    align: usize,
    can_overwrite: bool,
    pager: Option<Arc<dyn Pager>>,
}

impl<R> VmarMapOptions<R> {
    /// Creates a default set of options with the VMO and the memory access
    /// permissions.
    ///
    /// The VMO must have access rights that correspond to the memory
    /// access permissions. For example, if `perms` contains `VmPerm::Write`,
    /// then `vmo.rights()` should contain `Rights::WRITE`.
    pub fn new(parent: Vmar<R>, size: usize, perms: VmPerms) -> Self {
        Self {
            parent,
            perms,
            backend_offset: 0,
            size,
            offset: None,
            align: PAGE_SIZE,
            can_overwrite: false,
            pager: None,
        }
    }

    /// Sets the size of the mapping.
    ///
    /// The size of a mapping may not be equal to that of the VMO.
    /// For example, it is ok to create a mapping whose size is larger than
    /// that of the VMO, although one cannot read from or write to the
    /// part of the mapping that is not backed by the VMO.
    /// So you may wonder: what is the point of supporting such _oversized_
    /// mappings?  The reason is two-fold.
    /// 1. VMOs are resizable. So even if a mapping is backed by a VMO whose
    /// size is equal to that of the mapping initially, we cannot prevent
    /// the VMO from shrinking.
    /// 2. Mappings are not allowed to overlap by default. As a result,
    /// oversized mappings can serve as a placeholder to prevent future
    /// mappings from occupying some particular address ranges accidentally.
    ///
    /// The default value is the size of the VMO.
    pub fn size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Sets the mapping's alignment.
    ///
    /// The default value is the page size.
    ///
    /// The provided alignment must be a power of two and a multiple of the
    /// page size.
    pub fn align(mut self, align: usize) -> Self {
        self.align = align;
        self
    }

    /// Sets the mapping's offset inside the VMAR.
    ///
    /// The offset must satisfy the alignment requirement.
    /// Also, the mapping's range `[offset, offset + size)` must be within
    /// the VMAR.
    ///
    /// If not set, the system will choose an offset automatically.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Sets whether the mapping can overwrite existing mappings.
    ///
    /// The default value is false.
    ///
    /// If this option is set to true, then the `offset` option must be
    /// set.
    pub fn can_overwrite(mut self, can_overwrite: bool) -> Self {
        self.can_overwrite = can_overwrite;
        self
    }

    pub fn pager(mut self, pager: Arc<dyn Pager>) -> Self {
        self.pager = Some(pager);
        self
    }

    pub fn backend_offset(mut self, backend_offset: usize) -> Self {
        self.backend_offset = offset;
        self
    }

    /// Creates the mapping.
    ///
    /// All options will be checked at this point.
    ///
    /// On success, the virtual address of the new mapping is returned.
    pub fn build(self) -> Result<Vaddr> {
        self.check_options()?;
        let parent_vmar = self.parent.0.clone();
        let vm_mapping = Arc::new(VmMapping::build_mapping(self)?);
        let map_to_addr = vm_mapping.map_to_addr();
        parent_vmar.add_mapping(vm_mapping);
        Ok(map_to_addr)
    }

    /// check whether all options are valid
    fn check_options(&self) -> Result<()> {
        // check align
        debug_assert!(self.align % PAGE_SIZE == 0);
        debug_assert!(self.align.is_power_of_two());
        if self.align % PAGE_SIZE != 0 || !self.align.is_power_of_two() {
            return_errno_with_message!(Errno::EINVAL, "invalid align");
        }
        if let Some(offset) = self.offset {
            debug_assert!(offset % self.align == 0);
            if offset % self.align != 0 {
                return_errno_with_message!(Errno::EINVAL, "invalid offset");
            }
        }
        //self.check_perms()?;
        self.check_overwrite()?;
        Ok(())
    }

    // /// check whether the vmperm is subset of vmo rights
    // fn check_perms(&self) -> Result<()> {
    //     let perm_rights = Rights::from(self.perms);
    //     self.vmo.check_rights(perm_rights)
    // }

    /// check whether the vmo will overwrite with any existing vmo or vmar
    fn check_overwrite(&self) -> Result<()> {
        if self.can_overwrite {
            // if can_overwrite is set, the offset cannot be None
            debug_assert!(self.offset.is_some());
            if self.offset.is_none() {
                return_errno_with_message!(
                    Errno::EINVAL,
                    "offset can not be none when can overwrite is true"
                );
            }
        }
        if self.offset.is_none() {
            // if does not specify the offset, we assume the map can always find suitable free region.
            // FIXME: is this always true?
            return Ok(());
        }
        let offset = self.offset.unwrap();
        let map_range = offset..(offset + self.size);
        self.parent
            .0
            .check_vmo_overwrite(map_range, self.can_overwrite)
    }
}
