package se.ton.t210.domain;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PasswordSaltRepository extends JpaRepository<PasswordSalt, Long> {

   Optional<PasswordSalt> findByMemberId(Long memberId);
}
