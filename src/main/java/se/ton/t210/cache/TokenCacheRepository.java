package se.ton.t210.cache;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface TokenCacheRepository extends CrudRepository<TokenCache, String> {

    Optional<TokenCache> findByEmail(String email);
}
