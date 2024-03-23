// SPDX-License-Identifier: MPL-2.0

//! Virtual memory (VM).
//!
//! There are two primary VM abstractions:
//! * Virtual Memory Address Regions (VMARs) a type of capability that manages
//! user address spaces.
//! * Virtual Memory Objects (VMOs) are are a type of capability that
//! represents a set of memory pages.
//!
//! The concepts of VMARs and VMOs are originally introduced by
//! [Zircon](https://fuchsia.dev/fuchsia-src/reference/kernel_objects/vm_object).
//! As capabilities, the two abstractions are aligned with our goal
//! of everything-is-a-capability, although their specifications and
//! implementations in C/C++ cannot apply directly to Asterinas.
//! In Asterinas, VMARs and VMOs, as well as other capabilities, are implemented
//! as zero-cost capabilities.

use core::ops::Range;

use aster_frame::config::PAGE_SIZE;

pub mod page_fault_handler;
mod pager;
pub mod perms;
pub mod vmar;
pub mod vmo;

/// Get the page index range from the byte range.
pub fn get_page_idx_range(byte_range: &Range<usize>) -> Range<usize> {
    let start = byte_range.start.align_down(PAGE_SIZE);
    let end = byte_range.end.align_up(PAGE_SIZE);
    (start / PAGE_SIZE)..(end / PAGE_SIZE)
}

/// Aligns the given address range to the page size boundaries.
pub fn get_aligned_addr_range(addr_range: &Range<usize>) -> Range<usize> {
    let start = addr_range.start.align_down(PAGE_SIZE);
    let end = addr_range.end.align_up(PAGE_SIZE);
    (start)..(end)
}
