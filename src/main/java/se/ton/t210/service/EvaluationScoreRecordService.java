package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;

import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

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
            .map(EvaluationItem::getId)
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

    public ScoreResponse uploadScore(UploadScoreRequest request, Long memberId) {
        final List<EvaluationItemScoreRecord> records = request.records(memberId);
        evaluationItemScoreRecordRepository.saveAll(records);
        final MemberScore memberScore = new MemberScore(memberId, request.scoreSum());
        memberScoreRepository.save(memberScore);
        return new ScoreResponse(memberScore.getScore());
    }

    public List<RankResponse> rank(ApplicationType applicationType, int topNumber) {
        final Map<Long, Member> members = memberRepository.findAllByApplicationType(applicationType).stream()
            .collect(Collectors.toMap(Member::getId, v -> v));
        final List<MemberScore> scores = memberScoreRepository.findAllByMemberIdInAndCreatedAtOrderByScore(members.keySet(), LocalDate.now());
        if(members.size() < topNumber || scores.size() < topNumber) {
            throw new IllegalArgumentException("Need more data for ranking");
        }
        final List<MemberScore> rankScores = scores.subList(scores.size()-topNumber, scores.size());
        return RankResponse.listOf(rankScores, members);
    }
}
