package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.MemberPersonalInfoResponse;

import java.util.Optional;

public interface MemberRepository extends JpaRepository<Member, Long>, JpaSpecificationExecutor<Member> {

    boolean existsByEmail(String email);

    boolean existsByEmailAndPassword(String email, String password);

    Optional<Member> findByEmail(String email);

    Optional<Member> findByName(String name);

    int countByApplicationType(ApplicationType applicationType);

    MemberPersonalInfoResponse getMemberByEmail(String email);
}
