package se.ton.t210.domain;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface JudgingClassRepository extends JpaRepository<JudgingClass, Long> {

    List<JudgingClass> findAllByJudgingItemId(Long judgingItemId);
}
