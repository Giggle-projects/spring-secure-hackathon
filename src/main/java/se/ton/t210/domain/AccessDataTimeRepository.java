package se.ton.t210.domain;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;

import java.time.LocalDateTime;
import java.util.List;

public interface AccessDataTimeRepository extends JpaRepository<AccessDateTime, Long>, JpaSpecificationExecutor<AccessDateTime> {

    List<AccessDateTime> findAllByMemberId(Long memberId, Specification<AccessDateTime> spec, Pageable pageable);

    List<AccessDateTime> findAllByAccessTimeGreaterThan(LocalDateTime dateFrom);

    List<AccessDateTime> findAllByAccessTimeLessThan(LocalDateTime dateTo);

    List<AccessDateTime> findAllByAccessTimeGreaterThanAndAccessTimeLessThan(LocalDateTime dateFrom,
                                                                             LocalDateTime dateTo
    );

    List<AccessDateTime> findAllByAccessTimeBetween(LocalDateTime dateFrom, LocalDateTime dateTo);

    static Specification<AccessDateTime> greaterThanOrEqualsByTime(LocalDateTime dateFrom) {
        return (root, query, criteriaBuilder) -> criteriaBuilder.greaterThanOrEqualTo(root.get(AccessDateTime_.accessTime), dateFrom);
    }

    static Specification<AccessDateTime> lessThanOrEqualsByTime(LocalDateTime dateTo) {
        return (root, query, criteriaBuilder) -> criteriaBuilder.lessThanOrEqualTo(root.get(AccessDateTime_.accessTime), dateTo);
    }

//    static Specification<AccessDateTime> equalsMemberId(Long memberId) {
//        return (root, query, criteriaBuilder) -> criteriaBuilder.equal(root.get(AccessDateTime_.memberId), memberId)
//    }


}
