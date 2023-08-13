package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationItemRepository;
import se.ton.t210.domain.EvaluationItemScoreRecord;
import se.ton.t210.domain.EvaluationItemScoreRecordRepository;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.domain.MemberScore;
import se.ton.t210.domain.MemberScoreRepository;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.MonthlyScoresResponse;
import se.ton.t210.dto.RecordCountResponse;
import se.ton.t210.dto.ScoreResponse;
import se.ton.t210.dto.TopMonthlyScoresResponse;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import se.ton.t210.dto.UploadScoreRequest;

import static java.util.stream.Collectors.averagingInt;
import static java.util.stream.Collectors.groupingBy;

@Service
public class EvaluationScoreRecordService {

    @Autowired
    private MemberScoreRepository memberScoreRepository;

    @Autowired
    private MemberRepository memberRepository;

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
        for (Long judgingItemId : judgingItemIds) {
            final List<EvaluationItemScoreRecord> scores = evaluationItemScoreRecordRepository.findAllByEvaluationItemIdAndCreatedAt(judgingItemId, date);
            final int top50PScore = scores.get((scores.size() / 2) + 1).getScore();
            final int top30PScore = scores.get((scores.size() / 3 * 2) + 1).getScore();
            top50PScores.put(judgingItemId, (double) top50PScore);
            top30PScores.put(judgingItemId, (double) top30PScore);
        }
        return TopMonthlyScoresResponse.listOf(top50PScores, top30PScores);
    }

    public RecordCountResponse count(ApplicationType applicationType) {
        final Set<Long> evaluationIds = evaluationItemRepository.findAllByApplicationType(applicationType).stream()
            .map(it -> it.getId())
            .collect(Collectors.toSet());
        final int count = evaluationItemScoreRecordRepository.countByEvaluationItemIdIn(evaluationIds);
        return new RecordCountResponse(applicationType, count);
    }

    public ScoreResponse myScore(Long memberId, LocalDate date) {
        final List<MemberScore> scoresByMonth = memberScoreRepository.findAllByMemberIdAndCreatedAt(memberId, date);
        final int avgMonthScore = scoresByMonth.stream()
                .map(MemberScore::getScore)
                .collect(averagingInt(it-> it))
                .intValue();
        return new ScoreResponse(avgMonthScore);
    }

    public ScoreResponse uploadScore(UploadScoreRequest request, Long id, LocalDate now) {
        return null;
    }

    public List<RankResponse> rank(ApplicationType applicationType, int topNumber, LocalDate now) {
        final Set<Long> memberIds = memberRepository.findAllByApplicationType(applicationType).stream()
                .map(it -> it.getId())
                .collect(Collectors.toSet());
        final List<MemberScore> scores = memberScoreRepository.findAllByMemberIdInAndCreatedAtOrderByScore(memberIds, now);
        final List<MemberScore> rankScores = scores.subList(Math.max(scores.size() - 3, 0), scores.size());

    }
}
