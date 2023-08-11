package se.ton.t210.service;

import java.util.stream.Collectors;
import org.springframework.stereotype.Service;
import se.ton.t210.domain.*;
import se.ton.t210.dto.MonthlyScoresResponse;
import se.ton.t210.dto.TopMonthlyScoresResponse;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.averagingInt;
import static java.util.stream.Collectors.groupingBy;

@Service
public class ScoreRecordService {

    private final JudgingItemRepository judgingItemRepository;
    private final ScoreRecordRepository scoreRecordRepository;

    public ScoreRecordService(JudgingItemRepository judgingItemRepository, ScoreRecordRepository scoreRecordRepository) {
        this.judgingItemRepository = judgingItemRepository;
        this.scoreRecordRepository = scoreRecordRepository;
    }

    public MonthlyScoresResponse averageScoresByJudgingItem(Long memberId, LocalDate date) {
        final List<ScoreRecord> scores = scoreRecordRepository.findAllByMemberIdAndCreatedAt(memberId, date);
        final Map<Long, Double> averageScoresByJudgingItem = scores.stream()
            .collect(groupingBy(ScoreRecord::getJudgingId, averagingInt(ScoreRecord::getScore)));
        return MonthlyScoresResponse.listOf(averageScoresByJudgingItem);
    }

    public TopMonthlyScoresResponse averageAllScoresByJudgingItem(ApplicationType applicationType, LocalDate date) {
        final Map<Long, Double> top50PScoresByJudgingItem = new HashMap<>();
        final Map<Long, Double> top30PScoresByJudgingItem = new HashMap<>();
        final List<Long> judgingItemIds = judgingItemRepository.findAllByApplicationType(applicationType)
                .stream()
                .map(JudgingItem::getId)
                .collect(Collectors.toList());
        for(Long judgingItemId : judgingItemIds) {
            final List<ScoreRecord> scores = scoreRecordRepository.findAllByJudgingIdAndCreatedAt(judgingItemId, date);
            final int top50PScore = scores.get((scores.size()/2)+1).getScore();
            final int top30PScore = scores.get((scores.size()/3 * 2)+1).getScore();
            top50PScoresByJudgingItem.put(judgingItemId, (double) top50PScore);
            top30PScoresByJudgingItem.put(judgingItemId, (double) top30PScore);
        }
        return TopMonthlyScoresResponse.listOf(top50PScoresByJudgingItem, top30PScoresByJudgingItem);
    }
}
