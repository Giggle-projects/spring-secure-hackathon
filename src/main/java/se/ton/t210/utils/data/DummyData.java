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
        var testMember = memberRepository.save(new Member("test", "test@naver.com", "12345", ApplicationType.PoliceOfficerMale));
        members.add(testMember);
        for (var member : members) {
            records(member);
        }
    }

    private void createMembers(int number) {
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
    }

    private void createEvaluationItemTable() {
        itemTable = new HashMap<>();
        itemTable.put(ApplicationType.PoliceOfficerMale, createPoliceOfficerMaleEvaluationItem(ApplicationType.PoliceOfficerMale));
        itemTable.put(ApplicationType.PoliceOfficerFemale, createPoliceOfficerFeMaleEvaluationItem(ApplicationType.PoliceOfficerFemale));
        itemTable.put(ApplicationType.FireOfficerMale, createFireOfficerMaleEvaluationItem(ApplicationType.FireOfficerMale));
        itemTable.put(ApplicationType.FireOfficerFemale, createFireOfficerFeMaleEvaluationItem(ApplicationType.FireOfficerFemale));
        itemTable.put(ApplicationType.CorrectionalOfficerMale, createCorrectionalOfficerMaleEvaluationItem(ApplicationType.CorrectionalOfficerMale));
        itemTable.put(ApplicationType.CorrectionalOfficerFemale, createCorrectionalOfficerFeMaleEvaluationItem(ApplicationType.CorrectionalOfficerFemale));
    }

    @Transactional
    public List<EvaluationItem> createPoliceOfficerMaleEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> policeOfficerMaleData = getPoliceOfficeItemName(applicationType);
        evaluationItemRepository.saveAll(policeOfficerMaleData);
        final List<List<Float>> policeOfficerMaleSectionData = List.of(
                List.of(0F, 40F, 43F, 46F, 49F, 52F, 55F, 58F, 61F, 64F),
                List.of(0F, 16F, 22F, 28F, 34F, 40F, 46F, 51F, 56F, 61F),
                List.of(0F, 32F, 36F, 40F, 43F, 46F, 49F, 52F, 55F, 58F),
                List.of(0F, 8.68F, 8.46F, 8.26F, 8.05F, 7.84F, 7.63F, 7.42F, 7.21F, 7F),
                List.of(0F, 35F, 41F, 47F, 52F, 57F, 62F, 67F, 72F, 77F)
        );
        saveEvaluationSectionScore(policeOfficerMaleData, policeOfficerMaleSectionData);
        return policeOfficerMaleData;
    }

    @Transactional
    public List<EvaluationItem> createPoliceOfficerFeMaleEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> policeOfficerFeMaleData = getPoliceOfficeItemName(applicationType);
        evaluationItemRepository.saveAll(policeOfficerFeMaleData);
        final List<List<Float>> policeOfficeFeMaleEvaluationData = List.of(
                List.of(0F, 25F, 28F, 31F, 34F, 36F, 38F, 40F, 42F, 44F),
                List.of(0F, 7F, 10F, 13F, 16F, 19F, 22F, 25F, 28F, 31F),
                List.of(0F, 23F, 27F, 31F, 35F, 39F, 43F, 47F, 51F, 55F),
                List.of(0F, 10.15F, 9.91F, 9.67F, 9.43F, 9.19F, 8.95F, 8.71F, 8.47F, 8.23F),
                List.of(0F, 24F, 28F, 32F, 35F, 38F, 41F, 44F, 47F, 51F)
        );
        saveEvaluationSectionScore(policeOfficerFeMaleData, policeOfficeFeMaleEvaluationData);
        return policeOfficerFeMaleData;
    }

    @Transactional
    public List<EvaluationItem> createFireOfficerMaleEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> fireOfficerMaleData = getFireOfficeItemName(applicationType);
        evaluationItemRepository.saveAll(fireOfficerMaleData);
        final List<List<Float>> fireOfficerMaleSectionData = List.of(
                List.of(0F, 48.1F, 50.1F, 51.6F, 52.9F, 54.2F, 55.5F, 56.8F, 58.1F, 60F),
                List.of(0F, 232F, 237F, 240F, 243F, 246F, 250F, 255F, 258F, 263F),
                List.of(0F, 44F, 45F, 46F, 47F, 48F, 49F, 50F, 51F, 52F),
                List.of(0F, 60F, 62F, 64F, 68F, 72F, 75F, 76F, 77F, 78F),
                List.of(0F, 17.4F, 18.4F, 19.9F, 20.7F, 21.7F, 22.5F, 23.3F, 24.3F, 25.8F)
        );
        saveEvaluationSectionScore(fireOfficerMaleData, fireOfficerMaleSectionData);
        return fireOfficerMaleData;
    }

    @Transactional
    public List<EvaluationItem> createFireOfficerFeMaleEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> fireOfficerFeMaleData = getFireOfficeItemName(applicationType);
        evaluationItemRepository.saveAll(fireOfficerFeMaleData);
        final List<List<Float>> fireOfficerFeMaleSectionData = List.of(
                List.of(0F, 29.0F, 30.3F, 31.2F, 32.0F, 33.0F, 33.8F, 34.7F, 35.8F, 37.0F),
                List.of(0F, 165F, 169F, 173F, 177F, 181F, 185F, 189F, 194F, 199F),
                List.of(0F, 34F, 35F, 36F, 37F, 38F, 39F, 40F, 41F, 42F),
                List.of(0F, 29F, 31F, 32F, 34F, 37F, 40F, 41F, 42F, 43F),
                List.of(0F, 20.7F, 21.7F, 22.7F, 23.5F, 24.9F, 25.5F, 26.2F, 26.8F, 28.0F)
        );
        saveEvaluationSectionScore(fireOfficerFeMaleData, fireOfficerFeMaleSectionData);
        return fireOfficerFeMaleData;
    }

    @Transactional
    public List<EvaluationItem> createCorrectionalOfficerMaleEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> correctionalOfficerMaleData = getCorrectionalOfficeMaleItemName(applicationType);
        evaluationItemRepository.saveAll(correctionalOfficerMaleData);
        final List<List<Float>> correctionalOfficerMaleSectionData = List.of(
                List.of(0F, 130F, 140F, 150F, 160F, 170F, 180F, 190F, 200F, 210F),
                List.of(0F, 230F, 235F, 240F, 245F, 250F, 255F, 260F, 265F, 270F),
                List.of(0F, 35F, 38F, 41F, 44F, 47F, 50F, 53F, 56F, 59F),
                List.of(0F, 10.20F, 10.05F, 9.90F, 9.75F, 9.60F, 9.45F, 9.30F, 9.15F, 9.00F),
                List.of(0F, 10.00F, 9.40F, 9.20F, 9.00F, 8.40F, 8.20F, 8.00F, 7.40F, 7.20F)
        );
        saveEvaluationSectionScore(correctionalOfficerMaleData, correctionalOfficerMaleSectionData);
        return correctionalOfficerMaleData;
    }

    @Transactional
    public List<EvaluationItem> createCorrectionalOfficerFeMaleEvaluationItem(ApplicationType applicationType) {
        final List<EvaluationItem> correctionalOfficerFeMaleData = getCorrectionalOfficeFemaleItemName(applicationType);
        evaluationItemRepository.saveAll(correctionalOfficerFeMaleData);
        final List<List<Float>> correctionalOfficerFeMaleSectionData = List.of(
                List.of(0F, 85F, 90F, 95F, 100F, 105F, 110F, 115F, 120F, 125F),
                List.of(0F, 170F, 175F, 180F, 185F, 190F, 195F, 200F, 205F, 210F),
                List.of(0F, 30F, 33F, 36F, 39F, 42F, 45F, 48F, 51F, 54F),
                List.of(0F, 11.70F, 11.50F, 11.30F, 11.10F, 10.90F, 10.70F, 10.50F, 10.30F, 10.1F),
                List.of(0F, 7.20F, 7.05F, 6.50F, 6.35F, 6.20F, 6.05F, 5.50F, 5.35F, 5.20F)
        );
        saveEvaluationSectionScore(correctionalOfficerFeMaleData, correctionalOfficerFeMaleSectionData);
        return correctionalOfficerFeMaleData;
    }

    @Transactional
    public void saveEvaluationSectionScore(List<EvaluationItem> policeOfficeMaleData, List<List<Float>> policeOfficeMaleEvaluationData) {
        int index = 0;
        for (EvaluationItem item : policeOfficeMaleData) {
            evaluationScoreSectionRepository.saveAll(
                    List.of(
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(0), 1),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(1), 2),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(2), 3),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(3), 4),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(4), 5),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(5), 6),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(6), 7),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(7), 8),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(8), 9),
                            new EvaluationScoreSection(item.getId(), policeOfficeMaleEvaluationData.get(index).get(9), 10)
                    ));
            index++;
        }
    }

    private static List<EvaluationItem> getPoliceOfficeItemName(ApplicationType applicationType) {
        return List.of(
                new EvaluationItem(applicationType, "악력(kg)"),
                new EvaluationItem(applicationType, "팔굽혀피기(회/1분)"),
                new EvaluationItem(applicationType, "윗몸일으키기(회/1분)"),
                new EvaluationItem(applicationType, "50m 달리기(초)"),
                new EvaluationItem(applicationType, "왕복오래달리기(회)")
        );
    }

    private static List<EvaluationItem> getFireOfficeItemName(ApplicationType applicationType) {
        return List.of(
                new EvaluationItem(applicationType, "악력(kg)"),
                new EvaluationItem(applicationType, "제자리 멀리 뛰기(cm)"),
                new EvaluationItem(applicationType, "윗몸일으키기(회/1분)"),
                new EvaluationItem(applicationType, "왕복오래달리기(회)"),
                new EvaluationItem(applicationType, "앉아 윗몸 앞으로 굽히기(cm)")
        );
    }

    private static List<EvaluationItem> getCorrectionalOfficeMaleItemName(ApplicationType applicationType) {
        return List.of(
                new EvaluationItem(applicationType, "배근력(kg)"),
                new EvaluationItem(applicationType, "제자리 멀리 뛰기(cm)"),
                new EvaluationItem(applicationType, "윗몸일으키기(회/1분)"),
                new EvaluationItem(applicationType, "왕복 달리기(초)"),
                new EvaluationItem(applicationType, "2km 달리기(분)")
        );
    }

    private static List<EvaluationItem> getCorrectionalOfficeFemaleItemName(ApplicationType applicationType) {
        return List.of(
                new EvaluationItem(applicationType, "배근력(kg)"),
                new EvaluationItem(applicationType, "제자리 멀리 뛰기(cm)"),
                new EvaluationItem(applicationType, "윗몸일으키기(회/1분)"),
                new EvaluationItem(applicationType, "왕복 달리기(초)"),
                new EvaluationItem(applicationType, "1.2km 달리기(분)")
        );
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
