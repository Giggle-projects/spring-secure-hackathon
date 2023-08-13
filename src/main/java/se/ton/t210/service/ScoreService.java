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
import java.util.Optional;

@Service
public class ScoreService {

    private final MemberRepository memberRepository;
    private final EvaluationItemRepository evaluationItemRepository;
    private final MonthlyScoreItemRepository monthlyScoreItemRepository;
    private final MonthlyScoreRepository monthlyScoreRepository;

    @Autowired
    private EvaluationScoreSectionRepository evaluationScoreSectionRepository;

    public ScoreService(MemberRepository memberRepository, EvaluationItemRepository evaluationItemRepository, MonthlyScoreItemRepository monthlyScoreItemRepository, MonthlyScoreRepository monthlyScoreRepository) {
        this.memberRepository = memberRepository;
        this.evaluationItemRepository = evaluationItemRepository;
        this.monthlyScoreItemRepository = monthlyScoreItemRepository;
        this.monthlyScoreRepository = monthlyScoreRepository;
    }

    public RecordCountResponse count(ApplicationType applicationType) {
        final int recordCnt = monthlyScoreRepository.countByApplicationType(applicationType);
        return new RecordCountResponse(recordCnt);
    }

    public ScoreResponse score(Long memberId, LocalDate date) {
        final MonthlyScore monthlyScore = monthlyScoreRepository.findByMemberIdAndYearMonth(memberId, date)
            .orElseThrow();
        return new ScoreResponse(monthlyScore.getScore());
    }

    @Transactional
    public ScoreResponse update(Long memberId, List<EvaluationScoreRequest> request, LocalDate yearMonth) {
        int evaluationScoreSum = 0;
        for(EvaluationScoreRequest scoreInfo : request) {
            evaluationScoreSum += evaluationScore(scoreInfo.getEvaluationItemId(), scoreInfo.getScore()).getScore();
        }
        final Member member = memberRepository.findById(memberId).orElseThrow();
        monthlyScoreRepository.deleteAllByMemberIdAndYearMonth(member.getId(), yearMonth);
        monthlyScoreRepository.save(MonthlyScore.of(member, evaluationScoreSum));
        return new ScoreResponse(evaluationScoreSum);
    }

    public ScoreResponse evaluationScore(Long evaluationItemId, int score) {
        return new ScoreResponse(evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItemId).stream()
            .filter(it -> it.getSectionBaseScore() < score)
            .max(Comparator.comparingInt(EvaluationScoreSection::getScore))
            .map(EvaluationScoreSection::getScore)
            .orElse(0));
    }

    public List<RankResponse> rank(ApplicationType applicationType, int rankCnt, LocalDate date) {
        final PageRequest page = PageRequest.of(0, rankCnt, Sort.by(Sort.Order.desc("score")));
        final List<MonthlyScore> rankScores = monthlyScoreRepository.findAllByApplicationTypeAndYearMonth(applicationType, date, page);
        final List<RankResponse> rankResponses = new ArrayList<>();
        int rank = 1;
        for (var score : rankScores) {
            final Member member = memberRepository.findById(score.getMemberId()).orElseThrow();
            rankResponses.add(RankResponse.of(rank++, member, score));
        }
        return rankResponses;
    }
}
