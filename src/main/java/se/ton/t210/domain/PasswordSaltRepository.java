package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

public interface PasswordSaltRepository extends JpaRepository<PasswordSalt, Long> {

}
