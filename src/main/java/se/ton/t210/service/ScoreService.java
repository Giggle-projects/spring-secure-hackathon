package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Service
public class ScoreService {

    private final MemberRepository memberRepository;
    private final EvaluationItemRepository evaluationItemRepository;
    private final EvaluationItemScoreItemRepository evaluationItemScoreItemRepository;
    private final MonthlyScoreRepository monthlyScoreRepository;

    @Autowired
    private EvaluationScoreSectionRepository evaluationScoreSectionRepository;

    public ScoreService(MemberRepository memberRepository, EvaluationItemRepository evaluationItemRepository, EvaluationItemScoreItemRepository evaluationItemScoreItemRepository, MonthlyScoreRepository monthlyScoreRepository) {
        this.memberRepository = memberRepository;
        this.evaluationItemRepository = evaluationItemRepository;
        this.evaluationItemScoreItemRepository = evaluationItemScoreItemRepository;
        this.monthlyScoreRepository = monthlyScoreRepository;
    }

    public RecordCountResponse count(ApplicationType applicationType) {
        final int recordCnt = monthlyScoreRepository.countByApplicationType(applicationType);
        return new RecordCountResponse(recordCnt);
    }

    public ExpectScoreResponse score(Long memberId, LocalDate date) {
        final MonthlyScore monthlyScore = monthlyScoreRepository.findByMemberIdAndYearMonth(memberId, date).orElseThrow();
        final int expectedScore = 0; // TODO :: expectedScore
        final int expectedGrade = 0; // TODO :: expectedScore
        return new ExpectScoreResponse(monthlyScore.getScore(), expectedScore, expectedGrade);
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
        return evaluationScore(itemId, newItemScore.getScore());
    }

    private void updateMonthlyScore(Member member, int evaluationScoreSum, LocalDate yearMonth) {
        monthlyScoreRepository.deleteAllByMemberIdAndYearMonth(member.getId(), yearMonth);
        monthlyScoreRepository.save(MonthlyScore.of(member, evaluationScoreSum));
    }

    public int evaluationScore(Long evaluationItemId, int score) {
        return evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItemId).stream()
            .filter(it -> it.getSectionBaseScore() <= score)
            .max(Comparator.comparingInt(EvaluationScoreSection::getEvaluationScore))
            .map(EvaluationScoreSection::getEvaluationScore)
            .orElseThrow(() -> new IllegalArgumentException("Invalid evaluationItemId or score"));
    }

    public List<RankResponse> rank(ApplicationType applicationType, int rankCnt, LocalDate date) {
        final PageRequest page = PageRequest.of(0, rankCnt, Sort.by(Sort.Order.desc("score"), Sort.Order.asc("id")));
        final List<MonthlyScore> rankScores = monthlyScoreRepository.findAllByApplicationTypeAndYearMonth(applicationType, date, page);
        final List<RankResponse> rankResponses = new ArrayList<>();
        int rank = 0;
        int prevScore = Integer.MAX_VALUE;
        int sameStack = 0;
        for (var score : rankScores) {
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
}
