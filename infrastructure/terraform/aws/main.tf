terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "region" {
  type        = string
  description = "AWS region for future managed infrastructure."
  default     = "us-east-1"
}

provider "aws" { region = var.region }

# Intentionally no resources: provision only after a reviewed threat model,
# network design, cost estimate, and HIPAA-appropriate business agreements.
