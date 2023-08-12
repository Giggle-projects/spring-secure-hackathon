package se.ton.t210.service;

import java.util.stream.Collectors;
import org.springframework.stereotype.Service;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.MonthlyScoresResponse;
import se.ton.t210.dto.TopMonthlyScoresResponse;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.averagingInt;
import static java.util.stream.Collectors.groupingBy;

@Service
public class EvaluationScoreRecordService {

    private final EvaluationItemRepository evaluationItemRepository;
    private final EvaluationItemScoreRecordRepository evaluationItemScoreRecordRepository;

    public EvaluationScoreRecordService(EvaluationItemRepository evaluationItemRepository, EvaluationItemScoreRecordRepository evaluationItemScoreRecordRepository) {
        this.evaluationItemRepository = evaluationItemRepository;
        this.evaluationItemScoreRecordRepository = evaluationItemScoreRecordRepository;
    }

    public MonthlyScoresResponse averageScoresByJudgingItem(Long memberId, LocalDate date) {
        final List<EvaluationItemScoreRecord> scores = evaluationItemScoreRecordRepository.findAllByMemberIdAndCreatedAt(memberId, date);
        final Map<Long, Double> averageScoresByJudgingItem = scores.stream()
            .collect(groupingBy(EvaluationItemScoreRecord::getEvaluationItemId, averagingInt(EvaluationItemScoreRecord::getScore)));
        return MonthlyScoresResponse.listOf(averageScoresByJudgingItem);
    }

    public TopMonthlyScoresResponse averageAllScoresByJudgingItem(ApplicationType applicationType, LocalDate date) {
        final Map<Long, Double> top50PScores = new HashMap<>();
        final Map<Long, Double> top30PScores = new HashMap<>();
        final List<Long> judgingItemIds = evaluationItemRepository.findAllByApplicationType(applicationType)
                .stream()
                .map(EvaluationItem::getId)
                .collect(Collectors.toList());
        for(Long judgingItemId : judgingItemIds) {
            final List<EvaluationItemScoreRecord> scores = evaluationItemScoreRecordRepository.findAllByEvaluationItemIdAndCreatedAt(judgingItemId, date);
            final int top50PScore = scores.get((scores.size()/2)+1).getScore();
            final int top30PScore = scores.get((scores.size()/3 * 2)+1).getScore();
            top50PScores.put(judgingItemId, (double) top50PScore);
            top30PScores.put(judgingItemId, (double) top30PScore);
        }
        return TopMonthlyScoresResponse.listOf(top50PScores, top30PScores);
    }
}
