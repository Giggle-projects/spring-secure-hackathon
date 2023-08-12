package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;

public interface EvaluationItemScoreRecordRepository extends JpaRepository<EvaluationItemScoreRecord, Long> {

    List<EvaluationItemScoreRecord> findAllByMemberIdAndCreatedAt(Long memberId, LocalDate date);

    List<EvaluationItemScoreRecord> findAllByEvaluationItemIdAndCreatedAt(Long evaluationItemId, LocalDate date);
}
