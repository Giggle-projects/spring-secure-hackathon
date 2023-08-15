package se.ton.t210.utils.data;

import org.apache.commons.lang3.RandomStringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.service.EvaluationItemService;
import se.ton.t210.service.ScoreService;

import javax.annotation.PostConstruct;
import java.util.*;

@Profile("dev")
@Component
class DummyData {

    private static final Random RANDOM = new Random();

    @Autowired
    private MemberRepository memberRepository;

    @Autowired
    private MonthlyScoreRepository monthlyScoreRepository;

    @Autowired
    private EvaluationItemRepository evaluationItemRepository;

    @Autowired
    private EvaluationScoreSectionRepository evaluationScoreSectionRepository;

    @Autowired
    private EvaluationItemScoreItemRepository evaluationItemScoreItemRepository;

    @Autowired
    private EvaluationItemService evaluationItemService;

    @Autowired
    private ScoreService scoreService;

    private List<Member> members;
    private Map<ApplicationType, List<EvaluationItem>> itemTable;

    @PostConstruct
    public void create() {
        createMembers(100);
        createEvaluationItemTable();
        for (var member : members) {
            records(member);
        }
    }

    private List<Member> createMembers(int number) {
        members = new ArrayList<>();
        for (var applicationType : ApplicationType.values()) {
            for (int i = 0; i < number; i++) {
                final Member member = new Member(
                    RandomStringUtils.randomAlphabetic(5),
                    RandomStringUtils.randomAlphabetic(10) + "@gmail.com",
                    "12345",
                    applicationType
                );
                members.add(member);
            }
        }
        memberRepository.saveAll(members);
        return members;
    }

    private Map<ApplicationType, List<EvaluationItem>> createEvaluationItemTable() {
        itemTable = new HashMap<>();
        for (var applicationType : ApplicationType.values()) {
            itemTable.put(applicationType, createDummyEvaluationItem(applicationType));
        }
        return itemTable;
    }

    @Transactional
    public List<EvaluationItem> createDummyEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> evaluationItems = List.of(
                new EvaluationItem(applicationType, "A"),
                new EvaluationItem(applicationType, "B"),
                new EvaluationItem(applicationType, "C"),
                new EvaluationItem(applicationType, "D"),
                new EvaluationItem(applicationType, "E")
        );
        evaluationItemRepository.saveAll(evaluationItems);
        for (EvaluationItem item : evaluationItems) {
            evaluationScoreSectionRepository.saveAll(List.of(
                    new EvaluationScoreSection(item.getId(), 0, 1),
                    new EvaluationScoreSection(item.getId(), 40, 2),
                    new EvaluationScoreSection(item.getId(), 43, 3),
                    new EvaluationScoreSection(item.getId(), 46, 4),
                    new EvaluationScoreSection(item.getId(), 49, 5),
                    new EvaluationScoreSection(item.getId(), 52, 6),
                    new EvaluationScoreSection(item.getId(), 55, 7),
                    new EvaluationScoreSection(item.getId(), 61, 8),
                    new EvaluationScoreSection(item.getId(), 64, 9),
                    new EvaluationScoreSection(item.getId(), 68, 10)
            ));
        }
        return evaluationItems;
    }

    @Transactional
    public void records(Member member) {
        final List<EvaluationItemScore> evaluationItemScores = new ArrayList<>();
        for (var evaluationItem : itemTable.get(member.getApplicationType())) {
            var itemScore = new EvaluationItemScore(member.getId(), evaluationItem.getId(), RANDOM.nextInt(100));
            evaluationItemScoreItemRepository.save(itemScore);
            evaluationItemScores.add(itemScore);
        }
        int evaluationScoreSum = 0;
        for (EvaluationItemScore itemScore : evaluationItemScores) {
            var score = scoreService.evaluate(itemScore.getEvaluationItemId(), itemScore.getScore());
            evaluationScoreSum += score;
        }
        monthlyScoreRepository.save(MonthlyScore.of(member, evaluationScoreSum));
    }
}
