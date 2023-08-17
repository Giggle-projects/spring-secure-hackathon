package se.ton.t210.service;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.*;
import se.ton.t210.dto.AccessDateTimeFilter;
import se.ton.t210.dto.AccessDateTimeResponse;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class AdminService {

    private final MemberRepository memberRepository;
    private final MonthlyScoreRepository monthlyScoreRepository;
    private final AccessDataTimeRepository accessDataTimeRepository;

    public AdminService(MemberRepository memberRepository, MonthlyScoreRepository monthlyScoreRepository, AccessDataTimeRepository accessDataTimeRepository) {
        this.memberRepository = memberRepository;
        this.monthlyScoreRepository = monthlyScoreRepository;
        this.accessDataTimeRepository = accessDataTimeRepository;
    }

    @Transactional
    public List<AccessDateTimeResponse> findAll(Pageable pageable, AccessDateTimeFilter filter) {
        final LocalDateTime dateTimeFrom = filter.getDateFrom().orElse(LocalDate.MIN)
            .atTime(0, 0, 0);
        final LocalDateTime dateTimeTo = filter.getDateTo().orElse(LocalDate.MAX)
            .atTime(23, 59, 59);
        return accessDataTimeRepository.findAllByAccessTimeBetween(dateTimeTo, dateTimeFrom)
            .stream()
            .map(it -> {
                final Member member = memberRepository.findById(it.getMemberId()).orElseThrow();
                return AccessDateTimeResponse.of(it, member);
            }).collect(Collectors.toList());
    }
}
