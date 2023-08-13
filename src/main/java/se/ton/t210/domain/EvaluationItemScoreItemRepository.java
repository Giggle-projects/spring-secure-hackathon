package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;

public interface EvaluationItemScoreItemRepository extends JpaRepository<EvaluationItemScore, Long> {

    void deleteAllByMemberIdAndEvaluationItemIdAndYearMonth(Long memberId, Long evaluationItemId, LocalDate yearMonth);
}
