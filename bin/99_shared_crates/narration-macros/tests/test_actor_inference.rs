//! Tests for actor inference from module paths.
//!
//! Verifies that the macros correctly infer the actor from the module path.

use observability_narration_macros::{narrate, trace_fn};

/// Test actor inference in orchestratord module
mod orchestratord {
    use super::*;

    #[test]
    fn test_narrate_in_orchestratord() {
        #[narrate(action = "test", human = "Testing in orchestratord")]
        fn test_function() -> String {
            "orchestratord".to_string()
        }

        assert_eq!(test_function(), "orchestratord");
    }

    #[test]
    fn test_trace_fn_in_orchestratord() {
        #[trace_fn]
        fn test_function() -> String {
            "orchestratord".to_string()
        }

        assert_eq!(test_function(), "orchestratord");
    }

    mod admission {
        use super::*;

        #[test]
        fn test_nested_module() {
            #[narrate(action = "test", human = "Testing in nested module")]
            fn nested_function() -> String {
                "nested".to_string()
            }

            assert_eq!(nested_function(), "nested");
        }
    }
}

/// Test actor inference in pool_managerd module
mod pool_managerd {
    use super::*;

    #[test]
    fn test_narrate_in_pool_managerd() {
        #[narrate(action = "test", human = "Testing in pool_managerd")]
        fn test_function() -> String {
            "pool_managerd".to_string()
        }

        assert_eq!(test_function(), "pool_managerd");
    }

    #[test]
    fn test_trace_fn_in_pool_managerd() {
        #[trace_fn]
        fn test_function() -> String {
            "pool_managerd".to_string()
        }

        assert_eq!(test_function(), "pool_managerd");
    }
}

/// Test actor inference in worker_orcd module
mod worker_orcd {
    use super::*;

    #[test]
    fn test_narrate_in_worker_orcd() {
        #[narrate(action = "test", human = "Testing in worker_orcd")]
        fn test_function() -> String {
            "worker_orcd".to_string()
        }

        assert_eq!(test_function(), "worker_orcd");
    }

    #[test]
    fn test_trace_fn_in_worker_orcd() {
        #[trace_fn]
        fn test_function() -> String {
            "worker_orcd".to_string()
        }

        assert_eq!(test_function(), "worker_orcd");
    }
}

/// Test actor inference in vram_residency module
mod vram_residency {
    use super::*;

    #[test]
    fn test_narrate_in_vram_residency() {
        #[narrate(action = "test", human = "Testing in vram_residency")]
        fn test_function() -> String {
            "vram_residency".to_string()
        }

        assert_eq!(test_function(), "vram_residency");
    }

    #[test]
    fn test_trace_fn_in_vram_residency() {
        #[trace_fn]
        fn test_function() -> String {
            "vram_residency".to_string()
        }

        assert_eq!(test_function(), "vram_residency");
    }
}

/// Test actor inference in unknown module (fallback to "unknown")
mod unknown_service {
    use super::*;

    #[test]
    fn test_narrate_in_unknown_module() {
        #[narrate(action = "test", human = "Testing in unknown module")]
        fn test_function() -> String {
            "unknown".to_string()
        }

        assert_eq!(test_function(), "unknown");
    }

    #[test]
    fn test_trace_fn_in_unknown_module() {
        #[trace_fn]
        fn test_function() -> String {
            "unknown".to_string()
        }

        assert_eq!(test_function(), "unknown");
    }
}

/// Test that actor inference works at different nesting levels
mod deeply {
    mod nested {
        mod orchestratord {
            mod submodule {
                use observability_narration_macros::narrate;

                #[test]
                fn test_deeply_nested() {
                    #[narrate(action = "test", human = "Testing deeply nested")]
                    fn deeply_nested_function() -> String {
                        "nested".to_string()
                    }

                    assert_eq!(deeply_nested_function(), "nested");
                }
            }
        }
    }
}

/// Test actor inference with multiple known services in path
mod llama_orch {
    mod orchestratord {
        mod pool_managerd {
            use observability_narration_macros::narrate;

            #[test]
            fn test_multiple_services_in_path() {
                // Should match the first known service found
                #[narrate(action = "test", human = "Testing multiple services")]
                fn multi_service_function() -> String {
                    "multi".to_string()
                }

                assert_eq!(multi_service_function(), "multi");
            }
        }
    }
}
