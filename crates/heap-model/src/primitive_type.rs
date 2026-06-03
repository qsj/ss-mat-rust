/// JVM primitive type codes used in HPROF files.
pub const TYPE_OBJECT: u8 = 2;
pub const TYPE_BOOLEAN: u8 = 4;
pub const TYPE_CHAR: u8 = 5;
pub const TYPE_FLOAT: u8 = 6;
pub const TYPE_DOUBLE: u8 = 7;
pub const TYPE_BYTE: u8 = 8;
pub const TYPE_SHORT: u8 = 9;
pub const TYPE_INT: u8 = 10;
pub const TYPE_LONG: u8 = 11;

/// Returns the size in bytes for a given JVM type code.
#[inline]
pub fn type_size(type_code: u8) -> usize {
    match type_code {
        TYPE_OBJECT => 0, // size depends on pointer size, handled separately
        TYPE_BOOLEAN => 1,
        TYPE_CHAR => 2,
        TYPE_FLOAT => 4,
        TYPE_DOUBLE => 8,
        TYPE_BYTE => 1,
        TYPE_SHORT => 2,
        TYPE_INT => 4,
        TYPE_LONG => 8,
        _ => 0,
    }
}

/// Returns the type name for a given JVM type code.
pub fn type_name(type_code: u8) -> &'static str {
    match type_code {
        TYPE_OBJECT => "Object",
        TYPE_BOOLEAN => "boolean",
        TYPE_CHAR => "char",
        TYPE_FLOAT => "float",
        TYPE_DOUBLE => "double",
        TYPE_BYTE => "byte",
        TYPE_SHORT => "short",
        TYPE_INT => "int",
        TYPE_LONG => "long",
        _ => "unknown",
    }
}
