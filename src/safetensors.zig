const std = @import("std");

const tensor = @import("tensor.zig");

/// ## Semantics
/// - `path` is not consumed
/// - The returned data must be freed by the caller
fn read_bytes(path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.fs.cwd().createFile(
        path,
        .{ .read = true, .truncate = false },
    );
    const stat = try file.stat();
    const buffer = try allocator.alloc(u8, stat.size);
    try file.seekTo(0);
    const bytes_read = try file.readAll(buffer);
    std.debug.assert(bytes_read == stat.size);

    return buffer;
}

/// Read the first 8 bytes as the header length and return that as a slice of the original data.
///
/// Also returns the new offset.
///
/// ## Semantics
/// - `ct` is not consumed
/// - The returned data is a slice of ct
fn get_header_slice(ct: []u8) struct { []u8, u64 } {
    var length: u64 = 0;
    for (0..8) |index| {
        length |= (@as(u64, ct[index])) << (@as(u6, @intCast(index)) * @as(u6, 8));
    }
    const offset: usize = @intCast(length);
    return .{ ct[8 .. 8 + offset], 8 + offset };
}

const SafetensorsTensorMeta = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const u64,
    data_offsets: []const u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *const SafetensorsTensorMeta) void {
        self.allocator.free(self.name);
        self.allocator.free(self.dtype);
        self.allocator.free(self.shape);
        self.allocator.free(self.data_offsets);
        return;
    }
};

/// Parse the header.
///
/// ## Semantics
/// - `header` is not consumed
/// - The returned ArrayList must be freed with `.deinit`!
/// - The nested elements (`SafetensorsTensorMeta`) must each also have `.deinit` called!
fn parse_header(header: []u8, allocator: std.mem.Allocator) !std.ArrayList(SafetensorsTensorMeta) {
    const delim = "\":{";
    var splits = std.mem.split(u8, header, delim);
    var accum_metadata = std.ArrayList(SafetensorsTensorMeta).init(allocator);

    const Meta = struct {
        dtype: []const u8,
        shape: []const u64,
        data_offsets: []const u64,
    };

    var pos: usize = 0;
    while (splits.next()) |split| {
        if (std.mem.indexOf(u8, split, "dtype") == null) {
            pos += split.len + delim.len;
            continue;
        }

        if (std.mem.lastIndexOf(u8, split, "},\"") != null) {
            const idx = std.mem.lastIndexOf(u8, split, "},\"").?;
            const split_slice = split[0..idx];

            const buf = try allocator.alloc(u8, split_slice.len + 16);
            defer allocator.free(buf);
            const formatted = try std.fmt.bufPrint(buf, "{{ {s} }}", .{split_slice});
            const parsed = std.json.parseFromSlice(
                Meta,
                allocator,
                formatted,
                .{ .allocate = .alloc_always },
            ) catch {
                return error.JsonParseError;
            };
            defer parsed.deinit();

            // Name
            const pos_start = std.mem.lastIndexOf(u8, header[0..pos], "\"").?;
            var posn = pos_start - 1;
            while (header[posn] != '\"') {
                posn -= 1;
            }
            const name = header[posn + 1 .. pos];
            const end_quote_pos = std.mem.lastIndexOf(u8, name, "\"").?;

            const final_name = try allocator.alloc(u8, end_quote_pos);
            @memcpy(final_name, name[0..end_quote_pos]);

            // Other metadata
            const final_dtype = try allocator.alloc(u8, parsed.value.dtype.len);
            @memcpy(final_dtype, parsed.value.dtype);
            const final_shape: []u64 = try allocator.alloc(u64, parsed.value.shape.len);
            @memcpy(final_shape, parsed.value.shape);
            const final_offsets: []u64 = try allocator.alloc(u64, parsed.value.data_offsets.len);
            @memcpy(final_offsets, parsed.value.data_offsets);
            try accum_metadata.append(SafetensorsTensorMeta{ .name = final_name, .dtype = final_dtype, .data_offsets = final_offsets, .shape = final_shape, .allocator = allocator });
        } else if (std.mem.lastIndexOf(u8, split, "}}") != null) {
            const idx = std.mem.lastIndexOf(u8, split, "}}").?;
            const split_slice = split[0..idx];

            const buf = try allocator.alloc(u8, split_slice.len + 16);
            defer allocator.free(buf);
            const formatted = try std.fmt.bufPrint(buf, "{{ {s} }}", .{split_slice});
            const parsed = std.json.parseFromSlice(
                Meta,
                allocator,
                formatted,
                .{ .allocate = .alloc_always },
            ) catch {
                return error.JsonParseError;
            };
            defer parsed.deinit();

            // Name
            const pos_start = std.mem.lastIndexOf(u8, header[0..pos], "\"").?;
            var posn = pos_start - 1;
            while (header[posn] != '\"') {
                posn -= 1;
            }
            const name = header[posn + 1 .. pos];
            const end_quote_pos = std.mem.lastIndexOf(u8, name, "\"").?;

            const final_name = try allocator.alloc(u8, end_quote_pos);
            @memcpy(final_name, name[0..end_quote_pos]);

            // Other metadata
            const final_dtype = try allocator.alloc(u8, parsed.value.dtype.len);
            @memcpy(final_dtype, parsed.value.dtype);
            const final_shape: []u64 = try allocator.alloc(u64, parsed.value.shape.len);
            @memcpy(final_shape, parsed.value.shape);
            const final_offsets: []u64 = try allocator.alloc(u64, parsed.value.data_offsets.len);
            @memcpy(final_offsets, parsed.value.data_offsets);
            try accum_metadata.append(SafetensorsTensorMeta{ .name = final_name, .dtype = final_dtype, .data_offsets = final_offsets, .shape = final_shape, .allocator = allocator });
        }

        pos += split.len + delim.len;
    }
    return accum_metadata;
}

fn get_dtype(dtype: []const u8) error{UnknownDtype}!type {
    if (std.mem.eql(u8, dtype, "BOOL")) {
        return bool;
    } else if (std.mem.eql(u8, dtype, "U8")) {
        return u8;
    } else if (std.mem.eql(u8, dtype, "I8")) {
        return i8;
    } else if (std.mem.eql(u8, dtype, "I16")) {
        return i16;
    } else if (std.mem.eql(u8, dtype, "U16")) {
        return i16;
    } else if (std.mem.eql(u8, dtype, "F16")) {
        return f16;
    } else if (std.mem.eql(u8, dtype, "I32")) {
        return i32;
    } else if (std.mem.eql(u8, dtype, "U32")) {
        return u32;
    } else if (std.mem.eql(u8, dtype, "F32")) {
        return f32;
    } else if (std.mem.eql(u8, dtype, "F64")) {
        return f64;
    } else if (std.mem.eql(u8, dtype, "U64")) {
        return u64;
    } else {
        return error.UnknownDtype;
    }
}

/// Load tensors with names.
///
/// ## Semantics
/// - `path` is not consumed
/// - The returned ArrayList must be freed with `.deinit`!
/// - The nested tensors (`tensor.Tensor`) must each also have `.deinit` called!
pub fn load_tensors(comptime T: type, path: []const u8, allocator: std.mem.Allocator) !std.StringHashMap(tensor.Tensor(T)) {
    const bytes = try read_bytes(path, allocator);
    defer allocator.free(bytes);
    const header_slice = get_header_slice(bytes);
    const header: []u8 = header_slice[0];
    const data_offset: u64 = header_slice[1];

    const parsed_header = try parse_header(header, allocator);
    defer parsed_header.deinit();

    var accum = std.StringHashMap(tensor.Tensor(T)).init(allocator);
    for (parsed_header.items) |item| {
        defer item.deinit();
        const tensor_data = bytes[item.data_offsets[0] + data_offset .. item.data_offsets[1] + data_offset];
        if (!std.mem.eql(u8, item.dtype, "F32")) {
            std.debug.print("ERROR: unsupported dtype `{s}`.\n", .{item.dtype});
            std.debug.assert(false);
        }
        const ty = f32;
        const float_data = std.mem.bytesAsSlice(ty, @as([]align(@alignOf(ty)) u8, @alignCast(tensor_data)));
        const new_data = try allocator.alloc(ty, float_data.len);
        @memcpy(new_data, float_data);
        const name_cpy = try allocator.alloc(u8, item.name.len);
        @memcpy(name_cpy, item.name);
        const x = try tensor.Tensor(ty).from_slice(item.shape, new_data, allocator);
        try accum.put(name_cpy, x);
    }

    return accum;
}
