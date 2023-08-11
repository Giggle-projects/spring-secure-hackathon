package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface MemberRepository extends JpaRepository<Member, Long> {

    boolean existsByEmail(String email);

    boolean existsByEmailAndPassword(String email, String password);

    Optional<Member> findByEmail(String email);
}
