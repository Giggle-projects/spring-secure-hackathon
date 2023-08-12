package se.ton.t210.domain;

import java.util.List;
import java.util.Set;

import org.springframework.data.jpa.repository.JpaRepository;

public interface EvaluationScoreSectionRepository extends JpaRepository<EvaluationScoreSection, Long> {

    List<EvaluationScoreSection> findAllByEvaluationItemId(Long evaluationItemId);

    List<EvaluationScoreSection> findAllByEvaluationItemIdIn(Set<Long> evaluationItemIds);
}
