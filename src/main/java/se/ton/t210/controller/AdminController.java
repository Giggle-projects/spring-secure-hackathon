package se.ton.t210.controller;

import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.dto.AccessDateTimeFilter;
import se.ton.t210.dto.AccessDateTimeResponse;
import se.ton.t210.service.AdminService;

import java.util.List;

@RestController
public class AdminController {

    private final AdminService adminService;

    public AdminController(AdminService adminService) {
        this.adminService = adminService;
    }

    @GetMapping("/api/admin/users/access")
    public ResponseEntity<List<AccessDateTimeResponse>> usersMember(Pageable pageable, AccessDateTimeFilter filter) {
        final List<AccessDateTimeResponse> responses = adminService.findAll(pageable, filter);
        return ResponseEntity.ok(responses);
    }

    @PostMapping("/api/admin/block/users")
    public ResponseEntity<Void> blockMember(Long memberId) {
        adminService.saveBlackList(memberId);
        return ResponseEntity.ok().build();
    }
}
