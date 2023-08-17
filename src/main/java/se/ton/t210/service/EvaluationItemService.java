package se.ton.t210.service;

import org.springframework.stereotype.Service;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationItemRepository;
import se.ton.t210.domain.EvaluationScoreSection;
import se.ton.t210.domain.EvaluationScoreSectionRepository;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.ApplicationTypeInfoResponse;
import se.ton.t210.dto.EvaluationItemResponse;
import se.ton.t210.dto.EvaluationSectionInfo;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Service
public class EvaluationItemService {

    private final EvaluationScoreSectionRepository evaluationScoreSectionRepository;
    private final EvaluationItemRepository evaluationItemRepository;

    public EvaluationItemService(EvaluationScoreSectionRepository evaluationScoreSectionRepository, EvaluationItemRepository evaluationItemRepository) {
        this.evaluationScoreSectionRepository = evaluationScoreSectionRepository;
        this.evaluationItemRepository = evaluationItemRepository;
    }

    public List<EvaluationItemResponse> items(ApplicationType applicationType) {
        final List<EvaluationItem> items = evaluationItemRepository.findAllByApplicationType(applicationType);
        return EvaluationItemResponse.listOf(items);
    }

    public List<List<EvaluationSectionInfo>> sectionInfos(ApplicationType applicationType) {
        final List<List<EvaluationSectionInfo>> evaluationSectionsInfos = new ArrayList<>();
        final List<EvaluationItem> evaluationItems = evaluationItemRepository.findAllByApplicationType(applicationType);

        for (EvaluationItem evaluationItem : evaluationItems) {
            final List<EvaluationSectionInfo> evaluationSectionInfos = new ArrayList<>();
            final List<EvaluationScoreSection> sections = evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItem.getId());
            sections.sort(Comparator.comparingDouble(EvaluationScoreSection::getSectionBaseScore));

            boolean isScoreAsc = true;
            if(sections.size() > 1){
                isScoreAsc = sections.get(1).getSectionBaseScore() > sections.get(0).getSectionBaseScore();
            }
            if(isScoreAsc) {
                float prevItemBaseScore = Float.MAX_VALUE;
                for (EvaluationScoreSection section : sections) {
                    final EvaluationSectionInfo sectionInfo = EvaluationSectionInfo.of(evaluationItem, prevItemBaseScore, section);
                    evaluationSectionInfos.add(sectionInfo);
                    prevItemBaseScore = section.getSectionBaseScore() - 1;
                }
            } else {
                float prevItemBaseScore = 0f;
                for (EvaluationScoreSection section : sections) {
                    final EvaluationSectionInfo sectionInfo = EvaluationSectionInfo.of(evaluationItem, prevItemBaseScore, section);
                    evaluationSectionInfos.add(sectionInfo);
                    prevItemBaseScore = section.getSectionBaseScore() + 1;
                }
            }
            evaluationSectionsInfos.add(evaluationSectionInfos);
        }
        return evaluationSectionsInfos;
    }
}
