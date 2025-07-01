/// written fully by ai, not complicated, but didnt feel like spending the time to learn another
/// thing, just need a quick benchmark, seems simple to edit if i need to
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use moonlight::{ecs::EntityId, ecs::World};

/// Benchmark creating N entities
fn bench_entity_create(c: &mut Criterion) {
    c.bench_function("entity_create_10k", |b| {
        b.iter(|| {
            let mut world = World::init();
            for _ in 0..10_000 {
                world.entity_create();
            }
        })
    });
}

/// Benchmark adding a single component to N entities
fn bench_component_add(c: &mut Criterion) {
    // Initialize world and entities outside the timing loop to isolate component_add cost
    let mut world = World::init();
    let ids: Vec<EntityId> = (0..10_000).map(|_| world.entity_create()).collect();

    c.bench_function("component_add_10k", |b| {
        b.iter(|| {
            for &id in &ids {
                // Use black_box to prevent compiler optimizing away the value
                let _ = world.component_add(id, black_box((1u64, 2u64, 3u64)));
            }
        })
    });
}

/// Benchmark querying entities with the component
fn bench_query(c: &mut Criterion) {
    c.bench_function("query_10k", |b| {
        b.iter(|| {
            let mut world = World::init();
            let ids: Vec<EntityId> = (0..10_000).map(|_| world.entity_create()).collect();
            for &id in &ids {
                let _ = world.component_add(id, (1u64, 2u64, 3u64));
            }

            let results = world.query::<(u64, u64, u64)>();
            black_box(results);
        })
    });
}

/// Benchmark mutable query of two component types
fn bench_query2_mut(c: &mut Criterion) {
    let mut world = World::init();
    let ids: Vec<EntityId> = (0..10_000).map(|_| world.entity_create()).collect();
    for &id in &ids {
        world.component_add(id, 1u64).unwrap();
        world.component_add(id, 2u32).unwrap();
    }

    c.bench_function("query2_mut_10k", |b| {
        b.iter(|| {
            let results = world.query2_mut::<u64, u32>();
            black_box(results);
        })
    });
}

/// Benchmark removing component
fn bench_component_remove(c: &mut Criterion) {
    let mut world = World::init();
    let ids: Vec<EntityId> = (0..10_000).map(|_| world.entity_create()).collect();
    for &id in &ids {
        world.component_add(id, 42u64).unwrap();
    }

    c.bench_function("component_remove_10k", |b| {
        b.iter(|| {
            for &id in &ids {
                let _ = world.component_remove::<u64>(id);
            }
        })
    });
}

criterion_group!(
    ecs_benches,
    bench_entity_create,
    bench_component_add,
    bench_query,
    bench_query2_mut,
    bench_component_remove
);
criterion_main!(ecs_benches);
