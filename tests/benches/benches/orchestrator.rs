use atheer_orchestrator::{InferenceMode, Orchestrator, OrchestratorConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn orchestrator_creation(c: &mut Criterion) {
    c.bench_function("orchestrator_creation", |b| {
        b.iter(|| {
            let config = OrchestratorConfig::default();
            black_box(Orchestrator::new(config));
        })
    });
}

fn orchestrator_mode_switch(c: &mut Criterion) {
    let config = OrchestratorConfig::default();
    let mut orchestrator = Orchestrator::new(config);

    c.bench_function("orchestrator_mode_switch", |b| {
        b.iter(|| {
            orchestrator.set_mode(black_box(InferenceMode::Turbo));
            orchestrator.set_mode(black_box(InferenceMode::Eco));
            orchestrator.set_mode(black_box(InferenceMode::Balanced));
        })
    });
}

fn orchestrator_mode_query(c: &mut Criterion) {
    let config = OrchestratorConfig::default();
    let orchestrator = Orchestrator::new(config);

    c.bench_function("orchestrator_mode_query", |b| {
        b.iter(|| {
            black_box(orchestrator.current_mode());
            black_box(orchestrator.previous_mode());
        })
    });
}

criterion_group!(
    benches,
    orchestrator_creation,
    orchestrator_mode_switch,
    orchestrator_mode_query
);
criterion_main!(benches);
