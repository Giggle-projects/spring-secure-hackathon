package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;

public interface ScoreRecordRepository extends JpaRepository<ScoreRecord, Long> {

//    List<ScoreRecord> findAllByMemberIdAndCreatedAt_Month(Long memberId, LocalDate month);
//
//    List<ScoreRecord> findAllByJudgingIdAndCreatedAt_MonthOrderByScore(Long judgingId, LocalDate date);
//
    List<ScoreRecord> findAllByMemberId(Long memberId);

    List<ScoreRecord> findAllByJudgingIdOrderByScore(Long judgingItemId);
}
