package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Set;

public interface MonthlyScoreItemRepository extends JpaRepository<MonthlyEvaluationItemScore, Long> {

    List<MonthlyEvaluationItemScore> findAllByMemberIdAndYearMonth(Long memberId, LocalDate yearMonth);

    List<MonthlyEvaluationItemScore> findAllByEvaluationItemIdAndYearMonth(Long evaluationItemId, LocalDate yearMonth);

    int countByEvaluationItemIdIn(Set<Long> evaluationIds);
}
