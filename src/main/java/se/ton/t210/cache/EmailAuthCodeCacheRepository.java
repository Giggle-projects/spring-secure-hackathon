package se.ton.t210.cache;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface EmailAuthCodeCacheRepository extends CrudRepository<EmailAuthCodeCache, String> {
}
