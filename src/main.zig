const std = @import("std");

const tensor = @import("tensor.zig");

// Return a random number from 0 to given `max`
fn make_randn(seed: u64) f32 {
    var rng = std.rand.Isaac64.init(seed);
    return rng.random().float(f32);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 512;

    var a = try tensor.Tensor(f32).arange(0, N * N, allocator);
    defer a.deinit();

    a = try a.div_scalar(@floatFromInt(a.numel()));
    a = try a.view(&.{ N, N });

    var b = try tensor.Tensor(f32).arange(0, N * N, allocator);
    defer b.deinit();

    b = try b.div_scalar(@floatFromInt(b.numel()));
    b = try b.view(&.{ N, N });
    b = try b.t();

    const start1 = try std.time.Instant.now();
    const c1 = try a.matmul(&b);
    const ns_duration1 = @as(f32, @floatFromInt((try std.time.Instant.now()).since(start1))) / @as(f32, @floatFromInt(std.time.ns_per_ms));
    std.debug.print("matmul took {}ms\n", .{ns_duration1});

    std.debug.print("a: ", .{});
    std.debug.print("{any}\n", .{a.shape});
    // try a.debug();
    std.debug.print("b: ", .{});
    std.debug.print("{any}\n", .{b.shape});
    // try b.debug();
    std.debug.print("c1: ", .{});
    std.debug.print("{any}\n", .{c1.shape});
    // try c1.debug();
}
