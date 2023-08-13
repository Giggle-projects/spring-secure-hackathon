package se.ton.t210.service;

import org.springframework.stereotype.Service;
import se.ton.t210.domain.EvaluationItemRepository;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.domain.MonthlyScoreItemRepository;
import se.ton.t210.domain.MonthlyScoreRepository;

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
}
