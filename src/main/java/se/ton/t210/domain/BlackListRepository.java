package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import se.ton.t210.domain.type.ApplicationType;

import java.util.List;

public interface BlackListRepository extends JpaRepository<BlackList, Long> {

    boolean existsBlackListByMemberId(Long memberId);
}
