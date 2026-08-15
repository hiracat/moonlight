
struct RadianceLevelConfig {
    uvec3 grid_size;
    float grid_gap;
    /// the grid origin, varies between each level because how how each probe level is offset from
    /// the previous
    vec3  grid_origin;
    uint  sqrt_ray_count;
    float interval_start;
    float interval_end;
    uint  is_top_cascade;
    uint  _pad0;
}
