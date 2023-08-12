package se.ton.t210.controller;

import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.EvaluationScoreSection;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;
import se.ton.t210.dto.EvaluationItemNamesResponse;
import se.ton.t210.dto.EvaluationSectionInfo;
import se.ton.t210.dto.EvaluationSectionsInfos;
import se.ton.t210.dto.TakenScoreResponse;
import se.ton.t210.service.EvaluationItemService;

@RestController
public class EvaluationItemController {

    private final EvaluationItemService evaluationItemService;

    public EvaluationItemController(EvaluationItemService evaluationItemService) {
        this.evaluationItemService = evaluationItemService;
    }

    @GetMapping("/api/evaluation/score")
    public ResponseEntity<TakenScoreResponse> takenScore(Long evaluationItemId, int score) {
        final int takenScore = evaluationItemService.calculateTakenScore(evaluationItemId, score);
        return ResponseEntity.ok(TakenScoreResponse.of(takenScore));
    }

    @GetMapping("/api/evaluation/names")
    public ResponseEntity<List<EvaluationItemNamesResponse>> itemNames(ApplicationType applicationType) {
        final List<EvaluationItemNamesResponse> names = evaluationItemService.itemNames(applicationType);
        return ResponseEntity.ok(names);
    }

    @GetMapping("/api/evaluation/sections")
    public ResponseEntity<List<List<EvaluationSectionInfo>>> sections() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.PoliceOfficerMale);
        final List<List<EvaluationSectionInfo>> sectionInfos = evaluationItemService.getSectionInfos(member.getApplicationType());
        return ResponseEntity.ok(sectionInfos);
    }
}
