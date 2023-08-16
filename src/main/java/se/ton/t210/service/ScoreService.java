package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;
import se.ton.t210.utils.date.LocalDateUtils;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Transactional(readOnly = true)
@Service
public class ScoreService {

    private final MemberRepository memberRepository;
    private final EvaluationItemRepository evaluationItemRepository;
    private final EvaluationItemScoreItemRepository evaluationItemScoreItemRepository;
    private final MonthlyScoreRepository monthlyScoreRepository;
    private final EvaluationScoreSectionRepository evaluationScoreSectionRepository;

    public ScoreService(MemberRepository memberRepository,
        EvaluationItemRepository evaluationItemRepository,
        EvaluationItemScoreItemRepository evaluationItemScoreItemRepository,
        MonthlyScoreRepository monthlyScoreRepository,
        EvaluationScoreSectionRepository evaluationScoreSectionRepository) {
        this.memberRepository = memberRepository;
        this.evaluationItemRepository = evaluationItemRepository;
        this.evaluationItemScoreItemRepository = evaluationItemScoreItemRepository;
        this.monthlyScoreRepository = monthlyScoreRepository;
        this.evaluationScoreSectionRepository = evaluationScoreSectionRepository;
    }

    public ScoreCountResponse count(ApplicationType applicationType) {
        final int recordCnt = monthlyScoreRepository.countByApplicationType(applicationType);
        return new ScoreCountResponse(recordCnt);
    }

    public ExpectScoreResponse expect(Long memberId, ApplicationType applicationType, LocalDate yearMonth) {
        final MonthlyScore monthlyScore = monthlyScoreRepository.findByMemberIdAndYearMonth(memberId, yearMonth)
            .orElse(MonthlyScore.empty(applicationType, yearMonth));
        final int currentScore = monthlyScore.getScore();

        final int greaterThanMine = monthlyScoreRepository.countByApplicationTypeAndYearMonthAndScoreGreaterThan(applicationType, yearMonth, currentScore);
        final int totalCount = monthlyScoreRepository.countByApplicationTypeAndYearMonth(applicationType, yearMonth);
        final float currentPercentile = (float) (Math.floor(((float) greaterThanMine / totalCount * 100) * 100) / 100.0);

        final float expectedPassPercent = 0; // TODO :: expectedGrade
        return new ExpectScoreResponse(currentScore, currentPercentile, expectedPassPercent);
    }

    public MyScoreResponse myScores(Long memberId) {
        final List<MonthlyScore> scores = monthlyScoreRepository.findAllByMemberId(memberId);
        return MyScoreResponse.of(scores);
    }

    public List<EvaluationScoreByItemResponse> evaluationScores(Long memberId, ApplicationType applicationType, LocalDate yearMonth) {
        return evaluationItemRepository.findAllByApplicationType(applicationType).stream()
            .map(item -> {
                final float evaluationItemScore = evaluationItemScore(memberId, item, yearMonth);
                final int evaluationScore = evaluate(item.getId(), evaluationItemScore);
                return new EvaluationScoreByItemResponse(item.getId(), item.getName(), evaluationScore);
            }).collect(Collectors.toList());
    }

    private float evaluationItemScore(Long memberId, EvaluationItem item, LocalDate yearMonth) {
        return evaluationItemScoreItemRepository.findByEvaluationItemIdAndMemberIdAndYearMonth(item.getId(), memberId, yearMonth)
            .orElse(new EvaluationItemScore(memberId, item.getId(), 0))
            .getScore();
    }

    @Transactional
    public ScoreResponse update(Long memberId, List<EvaluationScoreRequest> request, LocalDate yearMonth) {
        int monthlyScore = 0;
        final Member member = memberRepository.findById(memberId).orElseThrow();
        for (EvaluationScoreRequest scoreInfo : request) {
            final int itemScore = updateEvaluationItemScore(memberId, yearMonth, scoreInfo);
            monthlyScore += itemScore;
        }
        updateMonthlyScore(member, monthlyScore, yearMonth);
        return new ScoreResponse(monthlyScore);
    }

    private int updateEvaluationItemScore(Long memberId, LocalDate yearMonth, EvaluationScoreRequest scoreInfo) {
        final Long itemId = scoreInfo.getEvaluationItemId();
        evaluationItemScoreItemRepository.deleteAllByMemberIdAndEvaluationItemIdAndYearMonth(memberId, itemId, yearMonth);
        final EvaluationItemScore newItemScore = evaluationItemScoreItemRepository.save(scoreInfo.toEntity(memberId));
        return evaluate(itemId, newItemScore.getScore());
    }

    private void updateMonthlyScore(Member member, int evaluationScoreSum, LocalDate yearMonth) {
        monthlyScoreRepository.deleteAllByMemberIdAndYearMonth(member.getId(), yearMonth);
        monthlyScoreRepository.save(MonthlyScore.of(member, evaluationScoreSum));
    }

    public int evaluate(Long evaluationItemId, float score) {
        return evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItemId).stream()
            .filter(it -> it.getSectionBaseScore() <= score)
            .max(Comparator.comparingInt(EvaluationScoreSection::getEvaluationScore))
            .map(EvaluationScoreSection::getEvaluationScore)
            .orElseThrow(() -> new IllegalArgumentException("Invalid evaluationItemId or score"));
    }

    public List<RankResponse> rank(ApplicationType applicationType, int rankCnt, LocalDate date) {
        final PageRequest page = PageRequest.of(0, rankCnt, Sort.by(Sort.Order.desc("score"), Sort.Order.asc("id")));
        final List<MonthlyScore> scores = monthlyScoreRepository.findAllByApplicationTypeAndYearMonth(applicationType, date, page);
        final List<RankResponse> rankResponses = new ArrayList<>();
        int rank = 0;
        int prevScore = Integer.MAX_VALUE;
        int sameStack = 0;
        for (var score : scores) {
            final Member member = memberRepository.findById(score.getMemberId()).orElseThrow();
            if (prevScore == score.getScore()) {
                sameStack++;
            } else {
                rank = rank + sameStack + 1;
                sameStack = 0;
            }
            prevScore = score.getScore();
            rankResponses.add(RankResponse.of(rank, member, score));
        }
        return rankResponses;
    }

    public List<MonthlyScoreResponse> scoresYear(LoginMemberInfo member, LocalDate year) {
        final ApplicationType applicationType = member.getApplicationType();
        return LocalDateUtils.monthsOfYear(year).stream()
            .map(yearMonth -> MonthlyScoreResponse.of(monthlyScoreRepository.findByMemberIdAndYearMonth(member.getId(), yearMonth)
                .orElse(MonthlyScore.empty(applicationType, yearMonth)))
            ).collect(Collectors.toList());
    }

    // avg evaluation scores of each evaluation item per monthly rankers
    public List<EvaluationScoreByItemResponse> avgEvaluationItemScoresTopOf(ApplicationType applicationType, int top, LocalDate yearMonth) {
        final Map<Long, List<EvaluationItemScore>> rankersScoresByItem = rankerScoresByItem(applicationType, top, yearMonth);
        final List<EvaluationScoreByItemResponse> responses = new ArrayList<>();
        for (Long itemId : rankersScoresByItem.keySet()) {
            final String itemName = evaluationItemRepository.findById(itemId).orElseThrow().getName();
            final double avgEvaluationScore = rankersScoresByItem.get(itemId).stream()
                .mapToInt(score -> evaluate(itemId, score.getScore()))
                .average()
                .orElseThrow();
            responses.add(new EvaluationScoreByItemResponse(itemId, itemName, (int) avgEvaluationScore));
        }
        return responses;
    }

    private Map<Long, List<EvaluationItemScore>> rankerScoresByItem(ApplicationType applicationType, int top, LocalDate yearMonth) {
        final List<Long> rankerIds = rankersByMonthlyScore(applicationType, top, yearMonth);
        return evaluationItemScoreItemRepository.findAllByMemberIdInAndYearMonth(rankerIds, yearMonth).stream()
            .collect(Collectors.groupingBy(EvaluationItemScore::getEvaluationItemId));
    }

    private List<Long> rankersByMonthlyScore(ApplicationType applicationType, int top, LocalDate yearMonth) {
        final int scoreCnt = monthlyScoreRepository.countByApplicationType(applicationType);
        final int fetchCount = (int) (scoreCnt * ((float) top / 100));
        final PageRequest page = PageRequest.of(0, fetchCount, Sort.by(Sort.Order.desc("score"), Sort.Order.asc("id")));
        return monthlyScoreRepository.findAllByApplicationTypeAndYearMonth(applicationType, yearMonth, page).stream()
            .map(MonthlyScore::getMemberId)
            .collect(Collectors.toList());
    }
}
