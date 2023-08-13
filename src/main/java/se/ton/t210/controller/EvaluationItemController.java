package se.ton.t210.controller;

import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;
import se.ton.t210.dto.*;
import se.ton.t210.service.EvaluationItemService;

@RestController
public class EvaluationItemController {

    private final EvaluationItemService evaluationItemService;

    public EvaluationItemController(EvaluationItemService evaluationItemService) {
        this.evaluationItemService = evaluationItemService;
    }

    @GetMapping("/api/evaluation/score")
    public ResponseEntity<ScoreResponse> evaluationScore(Long evaluationItemId, int memberInputScore) {
        final int evaluationScore = evaluationItemService.calculateEvaluationScore(evaluationItemId, memberInputScore);
        return ResponseEntity.ok(new ScoreResponse(evaluationScore));
    }

    @GetMapping("/api/evaluation/items")
    public ResponseEntity<List<EvaluationItemResponse>> items(ApplicationType applicationType) {
        final List<EvaluationItemResponse> responses = evaluationItemService.items(applicationType);
        return ResponseEntity.ok(responses);
    }

    @GetMapping("/api/evaluation/sections")
    public ResponseEntity<List<List<EvaluationSectionInfo>>> sections() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.PoliceOfficerMale);
        final List<List<EvaluationSectionInfo>> sectionInfos = evaluationItemService.sectionInfos(member.getApplicationType());
        return ResponseEntity.ok(sectionInfos);
    }
}
