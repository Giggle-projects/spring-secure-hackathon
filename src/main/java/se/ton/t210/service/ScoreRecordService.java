package se.ton.t210.service;

import org.springframework.stereotype.Service;
import se.ton.t210.domain.ScoreRecord;
import se.ton.t210.domain.ScoreRecordRepository;
import se.ton.t210.dto.MemberAverageScoresByJudgingItemResponse;

import java.time.LocalDate;
import java.time.Month;
import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.averagingInt;
import static java.util.stream.Collectors.groupingBy;

@Service
public class ScoreRecordService {

    private final ScoreRecordRepository scoreRecordRepository;

    public ScoreRecordService(ScoreRecordRepository scoreRecordRepository) {
        this.scoreRecordRepository = scoreRecordRepository;
    }

    public MemberAverageScoresByJudgingItemResponse averageScoresByJudgingItem(Long memberId, Month month) {
        final LocalDate localDate = LocalDate.of(LocalDate.now().getYear(), month, 0);
        final List<ScoreRecord> scores = scoreRecordRepository.findAllByMemberIdAndCreatedAt_Month(memberId, localDate);
        final Map<Long, Double> averageScoresByJudgingItem = scores.stream()
            .collect(groupingBy(ScoreRecord::getJudgingId, averagingInt(ScoreRecord::getScore)));
        return MemberAverageScoresByJudgingItemResponse.listOf(averageScoresByJudgingItem);
    }
}
