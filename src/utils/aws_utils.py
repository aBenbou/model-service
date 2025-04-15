# model-service/src/utils/aws_utils.py
def construct_s3_uri(bucket: str, prefix: str) -> str:
    """Construct an S3 URI from bucket and prefix"""
    return f"s3://{bucket}/{prefix}"


def is_s3_uri(path: str) -> bool:
    """Check if a path is an S3 URI"""
    return path.startswith("s3://")