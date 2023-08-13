package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Set;

public interface EvaluationItemScoreRecordRepository extends JpaRepository<EvaluationItemScoreRecord, Long> {

    List<EvaluationItemScoreRecord> findAllByMemberIdAndCreatedAt(Long memberId, LocalDate date);

    List<EvaluationItemScoreRecord> findAllByEvaluationItemIdAndCreatedAt(Long evaluationItemId, LocalDate date);

    int countByEvaluationItemIdIn(Set<Long> evaluationIds);
}
