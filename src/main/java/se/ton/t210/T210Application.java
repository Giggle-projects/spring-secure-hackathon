package se.ton.t210;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationItemRepository;
import se.ton.t210.domain.EvaluationScoreSection;
import se.ton.t210.domain.EvaluationScoreSectionRepository;
import se.ton.t210.domain.type.ApplicationType;

import java.util.List;

@ServletComponentScan
@SpringBootApplication
public class T210Application {

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(T210Application.class);
        app.setAdditionalProfiles("dev");
        final ConfigurableApplicationContext run = app.run(args);

        final DummyData bean = run.getBean(DummyData.class);
        bean.createDummy();
    }
}

@Component
class DummyData {

    @Autowired
    EvaluationItemRepository evaluationItemRepository;

    @Autowired
    EvaluationScoreSectionRepository evaluationScoreSectionRepository;

    @Transactional
    public void createDummy() {
        var ei1 = new EvaluationItem(ApplicationType.PoliceOfficerMale, "A");
        var ei2 = new EvaluationItem(ApplicationType.PoliceOfficerMale, "B");
        var ei3 = new EvaluationItem(ApplicationType.PoliceOfficerMale, "C");
        var ei4 = new EvaluationItem(ApplicationType.PoliceOfficerMale, "D");
        var ei5 = new EvaluationItem(ApplicationType.PoliceOfficerMale, "E");
        final List<EvaluationItem> eilist = List.of(ei1, ei2, ei3, ei4, ei5);
        evaluationItemRepository.saveAll(eilist);

        for(EvaluationItem ei : eilist) {
            var ess1 = new EvaluationScoreSection(ei.getId(), 0, 1);
            var ess2 = new EvaluationScoreSection(ei.getId(), 40, 2);
            var ess3 = new EvaluationScoreSection(ei.getId(), 43, 3);
            var ess4 = new EvaluationScoreSection(ei.getId(), 46, 4);
            var ess5 = new EvaluationScoreSection(ei.getId(), 49, 5);
            var ess6 = new EvaluationScoreSection(ei.getId(), 52, 5);
            var ess7 = new EvaluationScoreSection(ei.getId(), 55, 5);
            var ess8 = new EvaluationScoreSection(ei.getId(), 61, 5);
            var ess9 = new EvaluationScoreSection(ei.getId(), 64, 5);
            var ess10 = new EvaluationScoreSection(ei.getId(), 68, 5);
            evaluationScoreSectionRepository.saveAll(List.of(
                    ess1,
                    ess2,
                    ess3,
                    ess4,
                    ess5,
                    ess6,
                    ess7,
                    ess8,
                    ess9,
                    ess10
                )
            );
        }

    }
}
