package se.ton.t210.service;

import org.springframework.stereotype.Service;
import se.ton.t210.domain.EvaluationItem;
import se.ton.t210.domain.EvaluationItemRepository;
import se.ton.t210.domain.EvaluationScoreSection;
import se.ton.t210.domain.EvaluationScoreSectionRepository;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.EvaluationItemNamesResponse;

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

    public int calculateTakenScore(Long evaluationItemId, int score) {
        return evaluationScoreSectionRepository.findAllByEvaluationItemId(evaluationItemId).stream()
            .filter(it -> it.getSectionBaseScore() < score)
            .max(Comparator.comparingInt(EvaluationScoreSection::getScore))
            .map(EvaluationScoreSection::getScore)
            .orElse(0);
    }

    public List<EvaluationItemNamesResponse> itemNames(ApplicationType applicationType) {
        final List<EvaluationItem> items = evaluationItemRepository.findAllByApplicationType(applicationType);
        return EvaluationItemNamesResponse.listOf(items);
    }
}
