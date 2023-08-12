package se.ton.t210.domain;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface EvaluationScoreSectionRepository extends JpaRepository<EvaluationScoreSection, Long> {

    List<EvaluationScoreSection> findAllByEvaluationItemId(Long evaluationItemId);
}
