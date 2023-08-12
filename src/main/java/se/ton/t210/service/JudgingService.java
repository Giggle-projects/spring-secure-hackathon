package se.ton.t210.service;

import java.util.Comparator;
import java.util.List;
import org.springframework.stereotype.Service;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.JudgingClass;
import se.ton.t210.domain.JudgingClassRepository;
import se.ton.t210.domain.JudgingItem;
import se.ton.t210.domain.JudgingItemRepository;
import se.ton.t210.dto.JudgingItemNamesResponse;

@Service
public class JudgingService {

    private final JudgingClassRepository judgingClassRepository;
    private final JudgingItemRepository judgingItemRepository;

    public JudgingService(JudgingClassRepository judgingClassRepository, JudgingItemRepository judgingItemRepository) {
        this.judgingClassRepository = judgingClassRepository;
        this.judgingItemRepository = judgingItemRepository;
    }

    public int calculateTakenScore(Long judgingItemId, int score) {
        final List<JudgingClass> judgingClasses = judgingClassRepository.findAllByJudgingItemId(judgingItemId);
        return judgingClasses.stream()
                .filter(it -> it.getTargetScore() < score)
                .max(Comparator.comparingInt(JudgingClass::getTakenScore))
                .map(JudgingClass::getTakenScore).orElse(0);
    }

    public List<JudgingItemNamesResponse> itemNames(ApplicationType applicationType) {
        List<JudgingItem> items = judgingItemRepository.findAllByApplicationType(applicationType);
        return JudgingItemNamesResponse.listOf(items);
    }
}
