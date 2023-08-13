package se.ton.t210.utils.data;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.*;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;
import se.ton.t210.service.EvaluationScoreRecordService;

import java.time.LocalDate;
import java.util.List;

@Profile("dev")
@Component
class DummyData {

    @Autowired
    private MemberRepository memberRepository;

    @Autowired
    private EvaluationItemRepository evaluationItemRepository;

    @Autowired
    private EvaluationScoreSectionRepository evaluationScoreSectionRepository;

    @Autowired
    private EvaluationItemScoreRecordRepository evaluationItemScoreRecordRepository;

    @Autowired
    private EvaluationScoreRecordService evaluationScoreRecordService;

    public DummyData() {
        Member member = createDummyUser();
    }

    @Transactional
    public Member createDummyUser() {
        final var member = new Member("member", "dev@gmail.com", "12345", Gender.MALE, ApplicationType.PoliceOfficerMale, LocalDate.now(), LocalDate.now());
        memberRepository.save(member);

        final var evaluationItems = evaluationItemRepository.findAllByApplicationType(member.getApplicationType());

        var score = 10;
        for (EvaluationItem evaluationItem : evaluationItems) {
            evaluationItemScoreRecordRepository.save(new EvaluationItemScoreRecord(member.getId(), evaluationItem.getId(), score));
            score = score + 10;
        }
        return member;
    }

    @Transactional
    public void createDummyEvaluationItem(ApplicationType applicationType) {
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
                new EvaluationScoreSection(item.getId(), 52, 5),
                new EvaluationScoreSection(item.getId(), 55, 5),
                new EvaluationScoreSection(item.getId(), 61, 5),
                new EvaluationScoreSection(item.getId(), 64, 5),
                new EvaluationScoreSection(item.getId(), 68, 5)
            ));
        }
    }
}
