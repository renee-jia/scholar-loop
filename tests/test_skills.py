"""Skill Library: content-hash dedup, time-decay weighting, ranked render."""

from scholarloop.skills import Skill, SkillLibrary

DAY = 86400.0


def test_make_dedup_id_is_content_addressed():
    a = Skill.make("optimizer", 0.8, "use cosine schedule", "exp_1", 0.0)
    b = Skill.make("Optimizer", 0.5, "Use Cosine Schedule", "exp_2", 100.0)  # same content, diff case
    assert a.id == b.id


def test_weight_halves_each_half_life():
    s = Skill.make("c", 0.8, "m", "exp", 0.0)
    assert s.weight(0.0) == 0.8
    assert abs(s.weight(30 * DAY) - 0.4) < 1e-9
    assert abs(s.weight(60 * DAY) - 0.2) < 1e-9


def test_library_dedup_overwrites_same_lesson(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill.make("optimizer", 0.8, "cosine", "e1", 0.0))
    lib.add(Skill.make("optimizer", 0.9, "cosine", "e2", 100.0))   # same content -> one file
    assert len(lib.all()) == 1
    lib.add(Skill.make("arch", 0.5, "go deeper", "e3", 0.0))
    assert len(lib.all()) == 2


def test_skills_persist_and_isolate_per_domain(tmp_path):
    # a lesson written in one campaign is visible to a fresh library at the same per-domain path
    SkillLibrary.for_domain("digits-mlp", root=tmp_path).add(
        Skill.make("optimizer", 0.8, "cosine helps", "exp_1", 0.0))
    later = SkillLibrary.for_domain("digits-mlp", root=tmp_path)
    assert len(later.all()) == 1 and "cosine helps" in later.render(now=0.0)
    # a different domain is isolated — no cross-domain leakage
    assert SkillLibrary.for_domain("diabetes-mlp", root=tmp_path).all() == []


def test_active_ranks_by_decayed_weight(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill.make("a", 0.9, "recent strong", "e1", 10 * DAY))
    lib.add(Skill.make("b", 0.9, "old strong", "e2", 0.0))         # same severity, older -> lower weight
    rows = lib.active(now=10 * DAY, top_k=1)
    assert rows[0][0].mitigation == "recent strong"
    assert "recent strong" in lib.render(now=10 * DAY)


def test_relevance_query_surfaces_on_topic_lessons(tmp_path):
    lib = SkillLibrary(tmp_path)
    now = 1000.0
    # a heavy but off-topic lesson, and a lighter but on-topic one
    lib.add(Skill.make("optimizer", 0.9, "use cosine learning rate schedules for stability", "exp_1", now))
    lib.add(Skill.make("regularization", 0.4, "increase dropout to curb overfitting on small data", "exp_2", now))

    # no query -> pure weight order: the heavier (off-topic) lesson leads
    assert lib.active(now)[0][0].category == "optimizer"
    # a query about overfitting/dropout lifts the lighter on-topic lesson to the top
    ranked = lib.active(now, query="dropout overfitting small data regularization")
    assert ranked[0][0].category == "regularization"
    # render(query=...) reflects the same selection
    assert "dropout" in lib.render(now, top_k=1, query="dropout overfitting regularization")
