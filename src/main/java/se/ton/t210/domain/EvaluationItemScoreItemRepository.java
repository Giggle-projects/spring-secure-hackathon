package se.ton.t210.domain;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface EvaluationItemScoreItemRepository extends JpaRepository<EvaluationItemScore, Long> {

    void deleteAllByMemberIdAndEvaluationItemIdAndYearMonth(Long memberId, Long evaluationItemId, LocalDate yearMonth);

    Optional<EvaluationItemScore> findByEvaluationItemIdAndMemberIdAndYearMonth(Long evaluationItemId, Long memberId, LocalDate yearMonth);

    int countByEvaluationItemIdAndYearMonth(Long evaluationItemId, LocalDate yearMonth);

    List<EvaluationItemScore> findByEvaluationItemIdAndYearMonth(Long evaluationItemId, LocalDate yearMonth, PageRequest pageRequest);

    List<EvaluationItemScore> findAllByMemberIdInAndYearMonth(List<Long> membersIds, LocalDate yearMonth);

    List<EvaluationItemScore> findAllByMemberIdAndYearMonth(Long membersId, LocalDate yearMonth);
}
