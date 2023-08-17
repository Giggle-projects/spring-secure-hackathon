package se.ton.t210.service;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.*;
import se.ton.t210.dto.AccessDateTimeResponse;

import java.time.LocalDate;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AdminService {

    private final MemberRepository memberRepository;
    private final AccessDataTimeRepository accessDataTimeRepository;
    private final BlackListRepository blackListRepository;

    public AdminService(MemberRepository memberRepository, AccessDataTimeRepository accessDataTimeRepository, BlackListRepository blackListRepository) {
        this.memberRepository = memberRepository;
        this.accessDataTimeRepository = accessDataTimeRepository;
        this.blackListRepository = blackListRepository;
    }

    @Transactional
    public List<AccessDateTimeResponse> findAll() {
        return accessDataTimeRepository.findAll()
                .stream()
                .map(it -> {
                    final Member member = memberRepository.findById(it.getMemberId()).orElseThrow();
                    return AccessDateTimeResponse.of(it, member);
                }).collect(Collectors.toList());
    }

    @Transactional
    public void saveBlackList(Long memberId) {
        blackListRepository.save(new BlackList(memberId, LocalDate.now()));
    }
}
