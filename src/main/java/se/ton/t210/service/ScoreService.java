package se.ton.t210.service;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.RankResponse;
import se.ton.t210.dto.RecordCountResponse;
import se.ton.t210.dto.ScoreResponse;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@Service
public class ScoreService {

    private final MemberRepository memberRepository;
    private final EvaluationItemRepository evaluationItemRepository;
    private final MonthlyScoreItemRepository monthlyScoreItemRepository;
    private final MonthlyScoreRepository monthlyScoreRepository;

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
