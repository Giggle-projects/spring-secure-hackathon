package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.ApplicationTypeInfoResponse;
import se.ton.t210.dto.EvaluationItemResponse;
import se.ton.t210.dto.EvaluationSectionInfo;
import se.ton.t210.service.EvaluationItemService;

import java.util.List;

@RestController
public class EvaluationItemController {

    private final EvaluationItemService evaluationItemService;

    public EvaluationItemController(EvaluationItemService evaluationItemService) {
        this.evaluationItemService = evaluationItemService;
    }

    @GetMapping("/api/applicationType")
    public ResponseEntity<List<ApplicationTypeInfoResponse>> getApplicationType() {
        return ResponseEntity.ok(ApplicationTypeInfoResponse.listOf());
    }

    @GetMapping("/api/evaluation/items")
    public ResponseEntity<List<EvaluationItemResponse>> items() {
        final Member member = new Member(1L, "name", "email", "password", ApplicationType.PoliceOfficerMale);
        final List<EvaluationItemResponse> responses = evaluationItemService.items(member.getApplicationType());
        return ResponseEntity.ok(responses);
    }

    @GetMapping("/api/evaluation/sections")
    public ResponseEntity<List<List<EvaluationSectionInfo>>> sections() {
        final Member member = new Member(1L, "name", "email", "password", ApplicationType.PoliceOfficerMale);
        final List<List<EvaluationSectionInfo>> sectionInfos = evaluationItemService.sectionInfos(member.getApplicationType());
        return ResponseEntity.ok(sectionInfos);
    }
}
