package se.ton.t210.service;

import org.springframework.stereotype.Service;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationItemRepository;
import se.ton.t210.domain.EvaluationScoreSection;
import se.ton.t210.domain.EvaluationScoreSectionRepository;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.EvaluationItemResponse;
import se.ton.t210.dto.EvaluationSectionInfo;

import java.util.*;

@Service
public class EvaluationItemService {

    private final EvaluationScoreSectionRepository evaluationScoreSectionRepository;
    private final EvaluationItemRepository evaluationItemRepository;

    public EvaluationItemService(EvaluationScoreSectionRepository evaluationScoreSectionRepository, EvaluationItemRepository evaluationItemRepository) {
        this.evaluationScoreSectionRepository = evaluationScoreSectionRepository;
        this.evaluationItemRepository = evaluationItemRepository;
    }

    public int calculateEvaluationScore(Long evaluationItemId, int score) {
        return evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItemId).stream()
            .filter(it -> it.getSectionBaseScore() < score)
            .max(Comparator.comparingInt(EvaluationScoreSection::getScore))
            .map(EvaluationScoreSection::getScore)
            .orElse(0);
    }

    public List<EvaluationItemResponse> items(ApplicationType applicationType) {
        final List<EvaluationItem> items = evaluationItemRepository.findAllByApplicationType(applicationType);
        return EvaluationItemResponse.listOf(items);
    }

    public List<List<EvaluationSectionInfo>> sectionInfos(ApplicationType applicationType) {
        final List<List<EvaluationSectionInfo>> evaluationSectionsInfos = new ArrayList<>();
        final List<EvaluationItem> evaluationItems = evaluationItemRepository.findAllByApplicationType(applicationType);

        for(EvaluationItem evaluationItem : evaluationItems) {
            final List<EvaluationSectionInfo> evaluationSectionInfos = new ArrayList<>();
            final List<EvaluationScoreSection> sections = evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItem.getId());
            sections.sort(Comparator.comparingInt(EvaluationScoreSection::getSectionBaseScore));

            int prevItemBaseScore = 100;
            for(EvaluationScoreSection section : sections) {
                final EvaluationSectionInfo sectionInfo = new EvaluationSectionInfo(
                    evaluationItem.getId(),
                    evaluationItem.getName(),
                    prevItemBaseScore,
                    section.getSectionBaseScore(),
                    section.getScore()
                );
                evaluationSectionInfos.add(sectionInfo);
                prevItemBaseScore = section.getSectionBaseScore() -1;
            }
            evaluationSectionsInfos.add(evaluationSectionInfos);
        }
        return evaluationSectionsInfos;
    }
}
