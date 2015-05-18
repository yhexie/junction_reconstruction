#ifndef PCL_TYPES_H
#define PCL_TYPES_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

struct EIGEN_ALIGN16 _RichPoint
{
	EIGEN_ALIGN16 union
	{
		float data[4];
		struct
		{
			float x;
			float y;
			float z;
			float t;
		};
	};

	inline Eigen::Map<Eigen::Vector3f> getVector3fMap () { return (Eigen::Vector3f::Map (data)); }
	inline const Eigen::Map<const Eigen::Vector3f> getVector3fMap () const { return (Eigen::Vector3f::Map (data)); }
	inline Eigen::Map<Eigen::Vector4f, Eigen::Aligned> getVector4fMap () { return (Eigen::Vector4f::MapAligned (data)); }
	inline const Eigen::Map<const Eigen::Vector4f, Eigen::Aligned> getVector4fMap () const { return (Eigen::Vector4f::MapAligned (data)); }
	inline Eigen::Map<Eigen::Array3f> getArray3fMap () { return (Eigen::Array3f::Map (data)); }
	inline const Eigen::Map<const Eigen::Array3f> getArray3fMap () const { return (Eigen::Array3f::Map (data)); }
	inline Eigen::Map<Eigen::Array4f, Eigen::Aligned> getArray4fMap () { return (Eigen::Array4f::MapAligned (data)); }
	inline const Eigen::Map<const Eigen::Array4f, Eigen::Aligned> getArray4fMap () const { return (Eigen::Array4f::MapAligned (data)); }

	EIGEN_ALIGN16 union
	{
		float data_n[4];
		float normal[3];
		struct
		{
			float normal_x;
			float normal_y;
			float normal_z;
			float curvature;
		};
	};

	inline Eigen::Map<Eigen::Vector3f> getNormalVector3fMap () { return (Eigen::Vector3f::Map (data_n)); }
	inline const Eigen::Map<const Eigen::Vector3f> getNormalVector3fMap () const { return (Eigen::Vector3f::Map (data_n)); }
	inline Eigen::Map<Eigen::Vector4f, Eigen::Aligned> getNormalVector4fMap () { return (Eigen::Vector4f::MapAligned (data_n)); }
	inline const Eigen::Map<const Eigen::Vector4f, Eigen::Aligned> getNormalVector4fMap () const { return (Eigen::Vector4f::MapAligned (data_n)); }

	EIGEN_ALIGN16 union
	{
		float data_u[4];
		struct
		{
			int id_trajectory;
			int id_sample;
			float density;
			float car_id;
		};
	};
    
    
	EIGEN_ALIGN16 union
	{
		float data_u1[4];
		struct
		{
			int speed;
			int head;
			float lon;
			float lat;
		};
	};

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 RichPoint : public _RichPoint
{
	static const int ID_UNINITIALIZED = -1;
	static const int ID_FLOOR = 0;
	inline RichPoint ()
	{
		x = y = z = t = 0.0f;

		normal_x = normal_y = normal_z = curvature = 0.0f;

		id_trajectory = id_sample = ID_UNINITIALIZED;

		density = car_id = 0.0f;
        
        speed = head = 0;
        
        lon = lat = 0.0f;
	}

	inline RichPoint (const _RichPoint &p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		t = p.t;

		normal_x = p.normal_x;
		normal_y = p.normal_y;
		normal_z = p.normal_z;
		curvature = p.curvature;

		id_trajectory = p.id_trajectory;
		id_sample = p.id_sample;
		density = p.density;
		car_id = p.car_id;
        
        speed = p.speed;
        head = p.head;
        lon = p.lon;
        lat = p.lat;
	}

	template <class T>
	inline T castCoordinate(void) const
	{
		return T(x, y, z);
	}

	inline void setCoordinate(float _x, float _y, float _z)
	{
		x = _x; y = _y; z = _z;
	}

	template <class T>
	inline T castNormal(void) const
	{
		return T(normal_x, normal_y, normal_z);
	}

	inline void setNormal(float n_x, float n_y, float n_z)
	{
		normal_x = n_x;
		normal_y = n_y;
		normal_z = n_z;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::ostream& operator << (std::ostream& os, const RichPoint& p)
{
	os << "("
		<< p.x << "," << p.y << "," << p.z
		<< " - " << p.t
		<< " - " << p.normal_x << "," << p.normal_y << "," << p.normal_z
		<< " - " << p.curvature
		<< " - " << p.id_trajectory << "," << p.id_sample
		<< " - " << p.density
		<< " - " << p.car_id
        << " - " << p.speed
        << " - " << p.head
        << " - " << p.lon
        << " - " << p.lat
		<< ")";

	return (os);
}

POINT_CLOUD_REGISTER_POINT_STRUCT(RichPoint,
								  (float, x, x)
								  (float, y, y)
								  (float, z, z)
                                  (float, t, t)
								  (float, normal_x, normal_x)
								  (float, normal_y, normal_y)
								  (float, normal_z, normal_z)
								  (float, curvature, curvature)
								  (int, id_trajectory, id_trajectory)
								  (int, id_sample, id_sample)
								  (float, density, density)
								  (float, car_id, car_id)
                                  (int, speed, speed)
                                  (int, head, head)
                                  (float, lon, lon)
                                  (float, lat, lat)
								  )
POINT_CLOUD_REGISTER_POINT_WRAPPER(RichPoint, _RichPoint)

struct NullDeleter {void operator()(void const *) const {}};

#endif // PCL_TYPES_H
